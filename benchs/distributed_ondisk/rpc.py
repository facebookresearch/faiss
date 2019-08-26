# @nolint
# old code, not worthwhile to lint

"""
Simplistic RPC implementation.
Exposes all functions of a Server object.

Uses pickle for serialization and the socket interface.
"""

import os,pdb,pickle,time,errno,sys,_thread,traceback,socket,threading,gc


# default
PORT=12032


#########################################################################
# simple I/O functions



def inline_send_handle(f,conn):
  st=os.fstat(f.fileno())
  size=st.st_size
  pickle.dump(size,conn)
  conn.write(f.read(size))

def inline_send_string(s,conn):
  size=len(s)
  pickle.dump(size,conn)
  conn.write(s)


if False:

  def inline_send(filename,conn):
    inline_send_handle(open(filename,'r'),conn)

  def inline_recv_handle(f,conn):
    size=pickle.load(conn)
    rd=0
    while rd<size:
      sz=size-rd
      if sz>65536: sz=65536
      buf=conn.read(sz)
      f.write(buf)
      rd+=len(buf)

  def inline_recv(filename,conn):
    inline_recv_handle(open(filename,"w"),conn)



class FileSock:
  " wraps a socket so that it is usable by pickle/cPickle "

  def __init__(self,sock):
    self.sock=sock
    self.nr=0

  def write(self,buf):
    # print("sending %d bytes"%len(buf))
    #self.sock.sendall(buf)
    # print("...done")
    bs = 512 * 1024
    ns = 0
    while ns < len(buf):
      sent = self.sock.send(buf[ns:ns + bs])
      ns += sent


  def read(self,bs=512*1024):
    #if self.nr==10000: pdb.set_trace()
    self.nr+=1
    # print("read bs=%d"%bs)
    b = []
    nb = 0
    while len(b)<bs:
      # print('   loop')
      rb = self.sock.recv(bs - nb)
      if not rb: break
      b.append(rb)
      nb += len(rb)
    return b''.join(b)

  def readline(self):
    # print("readline!")
    """may be optimized..."""
    s=bytes()
    while True:
      c=self.read(1)
      s+=c
      if len(c)==0 or chr(c[0])=='\n':
        return s

class SocketTransferred(Exception):
  " the controlling socket of this RID has been given to another one "
  def __init__(self,to_rid):
    self.to_rid=to_rid

class CannotChangeRID(Exception):
  pass

class ClientExit(Exception):
  pass

class EndOfAsyncCall(Exception):
  pass

class ServerException(Exception):
  pass


class Server:
  """
  server protocol. Methods from classes that subclass Server can be called
  transparently from a client
  """

  instance_pool={}
  important_rids=[]
  instance_pool_lock=_thread.allocate_lock()

  def __init__(self,s,rid,datadir="data/",logf=sys.stderr):

    # base
    self.rid=rid
    self.logf=logf

    # register instance
    Server.instance_pool_lock.acquire()
    assert self.rid not in Server.instance_pool
    Server.instance_pool[self.rid]=self
    Server.instance_pool_lock.release()

    # connection

    self.conn=s
    self.fs=FileSock(s)

    # if data should be transmitted after the return message,
    # register a callback that does it here
    self.call_after_return=None

    # should the next function call be done in detached mode?
    # 0: no
    # 1,5: yes (at next function call)
    # 2,6: yes (at this function call)
    # if next_detach & 4: at end of function, we'll wait to
    #    return the result to an attach() call
    self.next_detach=0


  def log(self,s):
    self.logf.write("rid %d %s\n"%(self.rid,s))

  def get_by_rid(self,rid):
    Server.instance_pool_lock.acquire()
    # print "instance pool: ",Server.instance_pool.keys()
    try:
      other=Server.instance_pool[rid]
    except KeyError:
      other=None
    Server.instance_pool_lock.release()
    return other

  def resume_rid(self,rid):
    """ resumes a finished RID. Does NOT set the rid """
    Server.instance_pool_lock.acquire()
    try:
      # print "pool=%s, remove %d"%(Server.instance_pool,self.rid)
      if rid in Server.instance_pool:
        raise CannotChangeRID('RID busy')
      # silently move to new id
      del Server.instance_pool[self.rid]
      Server.instance_pool[rid]=self
    finally:
      Server.instance_pool_lock.release()

  def detach_at_next_call(self,join=True):
    """ next call will close the socket.
    If join, wait for attach() at end of function"""
    self.next_detach=join and 5 or 1

  def detach(self):
    self.det_lock=_thread.allocate_lock()
    self.det_lock.acquire()
    self.waiting_cond=None
    self.fs=None
    self.conn.close(); self.conn=None
    self.det_lock.release()

  def attach(self,rid):
    """ must be attached by *another* RID """

    other=self.get_by_rid(rid)
    if not other:
      raise CannotChangeRID("cannot find other")

    self.log("found other RID")

    other.det_lock.acquire()
    if not other.waiting_cond:
      other.det_lock.release()
      raise CannotChangeRID("other not in wait_attach")

    # transfer connection
    other.conn=self.conn;    self.conn=None
    other.fs=self.fs;        self.fs=None

    other.waiting_cond.notify()
    other.det_lock.release()

    # exit gracefully
    raise SocketTransferred(rid)

  def wait_attach(self):
    self.det_lock.acquire()
    self.waiting_cond=threading.Condition(self.det_lock)
    self.log("wait_attach wait")
    self.waiting_cond.wait()
    # shoud set timeout for dead clients
    self.log("wait_attach end of wait")
    self.det_lock.release()
    del self.waiting_cond
    del self.det_lock

  def one_function(self):
    """
    Executes a single function with associated I/O.
    Protocol:
    - the arguments and results are serialized with the pickle protocol
    - client sends : (fname,args)
        fname = method name to call
        args = tuple of arguments
    - server sends result: (rid,st,ret)
        rid = request id
        st = None, or exception if there was during execution
        ret = return value or None if st!=None
    - if data must be sent raw over the socket after the function returns,
      call_after_return must be set to a function that does this
    - if the client wants to close a connection during a function, it
      should call detach_at_next_call(join)
      If join then
        the client should reatach from another rid with attach(rid), which
        will return the function's result
      else
        the server exits at the end of the function call and drops the result
    """

    try:
      (fname,args)=pickle.load(self.fs)
    except EOFError:
      raise ClientExit("read args")
    self.log("executing method %s with args %s"%(fname,args))
    st=None
    ret=None
    try:
      f=getattr(self,fname)
    except AttributeError:
      st=AttributeError("unknown method "+fname)
      self.log("unknown method ")
    else:

      if self.next_detach in [2,6]:
        pickle.dump((self.rid,None,None),self.fs,protocol=4)
        self.detach()

      try:
        ret=f(*args)
      except SocketTransferred as e:
        raise e
      except CannotChangeRID:
        st="CannotChangeRID"
      except Exception as e:
        # due to a bug (in mod_python?), ServerException cannot be
        # unpickled, so send the string and make the exception on the client side

        #st=ServerException(
        #  "".join(traceback.format_tb(sys.exc_info()[2]))+
        #  str(e))
        st="".join(traceback.format_tb(sys.exc_info()[2]))+str(e)
        self.log("exception in method")
        traceback.print_exc(50,self.logf)
        self.logf.flush()

      if self.next_detach==6:
        self.wait_attach()
        self.next_detach=0
      if self.next_detach==2:
        raise EndOfAsyncCall()
      elif self.next_detach in [1,5]:
        self.next_detach+=1

    print("return",ret)
    # pdb.set_trace()
    try:
      pickle.dump((self.rid,st,ret),self.fs, protocol=4)
    except EOFError:
      raise ClientExit("function return")

    # pdb.set_trace()
    if self.call_after_return!=None:
      self.log("doing call_after_return")
      try:
        self.call_after_return()
      except Exception as e:
        # this is a problem: we can't propagate the exception to the client
        # so he will probably crash...
        self.log("!!! exception in call_after_return")
        traceback.print_exc(50,self.logf)
        traceback.print_exc(50,sys.stderr)
        self.logf.flush()
      self.call_after_return=None

  def exec_loop(self):
    """ main execution loop. Loops and handles exit states
    """

    self.log("in exec_loop")
    try:
      while True:
        self.one_function()
    except ClientExit as e:
      self.log("ClientExit %s"%e)
    except SocketTransferred as e:
      self.log("socket transferred to RID thread %d"%e.to_rid)
    except EndOfAsyncCall as e:
      self.log("EndOfAsyncCall %s"%e)
    except socket.error as e:
      self.log("socket error %s"%e)
      traceback.print_exc(50,self.logf)
    except EOFError:
      self.log("EOF during communication")
      traceback.print_exc(50,self.logf)
    except:
      # unexpected
      traceback.print_exc(50,sys.stderr)
      sys.exit(1)

    Server.instance_pool_lock.acquire()
    try:
      try:
        del Server.instance_pool[self.rid]
        self.exec_loop_cleanup()
      except:
        traceback.print_exc(50,sys.stderr)
    finally:
      Server.instance_pool_lock.release()

    print("exit rid %d "%self.rid)

  def exec_loop_cleanup(self):
    pass

  def get_datafile(self,fname):
    " sends a file from the datadir to the client "
    fh=open(fname,"rb")
    self.call_after_return=lambda *x: inline_send_handle(fh,self.fs)
    return None

  def put_datafile(self,fname):
    " puts a file coming from the client in the datadir "
    fname=self.datadir+fname
    fh=open(fname,"wb")
    self.call_after_return=lambda *x: inline_recv_handle(fh,self.fs)
    return fname

  ###################################################################
  # spying stuff

  def get_ps_stats(self):
    ret=''
    f=os.popen("echo ============ `hostname` uptime:; uptime;"+
               "echo ============ self:; "+
               "ps -p %d -o pid,vsize,rss,%%cpu,nlwp,psr; "%os.getpid()+
               "echo ============ run queue:;"+
               "ps ar -o user,pid,%cpu,%mem,ni,nlwp,psr,vsz,rss,cputime,command")
    for l in f:
      ret+=l
    return ret

  def get_server_pool(self):
    Server.instance_pool_lock.acquire()
    rids=list(Server.instance_pool.keys())
    Server.instance_pool_lock.release()
    return rids

  def i_am_important(self,key):
    Server.instance_pool_lock.acquire()
    Server.important_rids.append((self.rid,key))
    Server.instance_pool_lock.release()

  def get_importants(self):
    Server.instance_pool_lock.acquire()
    i=Server.important_rids[:]
    Server.instance_pool_lock.release()
    return i

class Client:
  """
  Methods of the server object can be called transparently. Exceptions are
  re-raised.
  """
  def __init__(self,HOST,port=PORT, v6=False):
    socktype = socket.AF_INET6 if v6 else socket.AF_INET

    sock = socket.socket(socktype, socket.SOCK_STREAM)
    print("connecting",HOST, port, socktype)
    sock.connect((HOST, port))
    self.sock=sock
    self.fs=FileSock(sock)

    self.rid=-1

    self.async_state=0

  def generic_fun(self,fname,args):
    # int "gen fun",fname
    if self.async_state==2:
      raise RuntimeError("async call in progress")

    pickle.dump((fname,args),self.fs, protocol=4)
    if self.async_state==1:
      self.async_state=2
    else:
      return self.get_result()

  def get_result(self):
    (rid,st,ret)=pickle.load(self.fs)
    self.async_state=0
    self.rid=rid
    if st!=None:
      if st=="CannotChangeRID":
        raise CannotChangeRID("")
      else:
        raise ServerException(st)
    else:
      return ret

  def async_at_next_call(self):
    self.async_state=1

  def __getattr__(self,name):
    return lambda *x: self.generic_fun(name,x)

  def get_datafile_handle(self,dist_name,local_fh):
    self.generic_fun('get_datafile',(dist_name,))
    inline_recv_handle(local_fh,self.fs)

  def put_datafile_handle(self,local_fh,dist_name):
    dist_name_2=self.generic_fun('put_datafile',(dist_name,))
    inline_send_handle(local_fh,self.fs)
    return dist_name_2

  def get_datafile(self,dist_name,local_name):
    return self.get_datafile_handle(dist_name,open(local_name,"wb"))

  def put_datafile(self,local_name,dist_name):
    return self.put_datafile_handle(open(local_name,"rb"),dist_name)


def run_server(new_handler, port=PORT, report_to_file=None, v6=False):

  HOST = ''                 # Symbolic name meaning the local host
  socktype = socket.AF_INET6 if v6 else socket.AF_INET
  s = socket.socket(socktype, socket.SOCK_STREAM)
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

  print("bind %s:%d" % (HOST, port))
  s.bind((HOST, port))
  s.listen(5)

  print("accepting connections")
  if report_to_file is not None:
    print('storing host+port in', report_to_file)
    open(report_to_file, 'w').write('%s:%d ' % (socket.gethostname(), port))


  while True:
    try:
      conn, addr = s.accept()
    except socket.error as e:
      if e[1]=='Interrupted system call': continue
      raise

    print('Connected by', addr, end=' ')

    ibs=new_handler(conn)

    print("handled by rid ",ibs.rid, end=' ')

    tid=_thread.start_new_thread(ibs.exec_loop,())

    print("tid",tid)
