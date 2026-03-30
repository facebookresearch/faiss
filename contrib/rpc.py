# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Simplistic RPC implementation.
Exposes all functions of a Server object.

This code is for demonstration purposes only, and does not include certain
security protections. It is not meant to be run on an untrusted network or
in a production environment.
"""

import importlib
import os
import pickle
import sys
import _thread
import traceback
import socket
import logging

LOG = logging.getLogger(__name__)

# default
PORT = 12032

safe_modules = {
    'numpy',
    'numpy.core.multiarray',
}


class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        # Only allow safe modules.
        if module in safe_modules:
            return getattr(importlib.import_module(module), name)
        # Forbid everything else.
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" %
                                     (module, name))


class FileSock:
    " wraps a socket so that it is usable by pickle/cPickle "

    def __init__(self,sock):
        self.sock = sock
        self.nr=0

    def write(self, buf):
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


class ClientExit(Exception):
    pass


class ServerException(Exception):
    pass


class Server:
    """
    server protocol. Methods from classes that subclass Server can be called
    transparently from a client
    """

    def __init__(self, s, logf=sys.stderr, log_prefix=''):
        self.logf = logf
        self.log_prefix = log_prefix

        # connection

        self.conn = s
        self.fs = FileSock(s)


    def log(self, s):
        self.logf.write("Server log %s: %s\n" % (self.log_prefix, s))

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
        """

        try:
            (fname, args) = RestrictedUnpickler(self.fs).load()
        except EOFError:
            raise ClientExit("read args")
        self.log("executing method %s"%(fname))
        st = None
        ret = None
        try:
            f=getattr(self,fname)
        except AttributeError:
            st = AttributeError("unknown method "+fname)
            self.log("unknown method")

        try:
            ret = f(*args)
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

        LOG.info("return")
        try:
            pickle.dump((st ,ret), self.fs, protocol=4)
        except EOFError:
            raise ClientExit("function return")

    def exec_loop(self):
        """ main execution loop. Loops and handles exit states"""

        self.log("in exec_loop")
        try:
            while True:
                self.one_function()
        except ClientExit as e:
            self.log("ClientExit %s"%e)
        except socket.error as e:
            self.log("socket error %s"%e)
            traceback.print_exc(50,self.logf)
        except EOFError:
            self.log("EOF during communication")
            traceback.print_exc(50,self.logf)
        except BaseException:
            # unexpected
            traceback.print_exc(50,sys.stderr)
            sys.exit(1)

        LOG.info("exit server")

    def exec_loop_cleanup(self):
        pass

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


class Client:
    """
    Methods of the server object can be called transparently. Exceptions are
    re-raised.
    """
    def __init__(self, HOST, port=PORT, v6=False):
        socktype = socket.AF_INET6 if v6 else socket.AF_INET

        sock = socket.socket(socktype, socket.SOCK_STREAM)
        LOG.info("connecting to %s:%d, socket type: %s", HOST, port, socktype)
        sock.connect((HOST, port))
        self.sock = sock
        self.fs = FileSock(sock)

    def generic_fun(self, fname, args):
        # int "gen fun",fname
        pickle.dump((fname, args), self.fs, protocol=4)
        return self.get_result()

    def get_result(self):
        (st, ret) = RestrictedUnpickler(self.fs).load()
        if st!=None:
            raise ServerException(st)
        else:
            return ret

    def __getattr__(self,name):
        return lambda *x: self.generic_fun(name,x)


def run_server(new_handler, port=PORT, report_to_file=None, v6=False):

    HOST = ''                 # Symbolic name meaning the local host
    socktype = socket.AF_INET6 if v6 else socket.AF_INET
    s = socket.socket(socktype, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    LOG.info("bind %s:%d", HOST, port)
    s.bind((HOST, port))
    s.listen(5)

    LOG.info("accepting connections")
    if report_to_file is not None:
        LOG.info('storing host+port in %s', report_to_file)
        open(report_to_file, 'w').write('%s:%d ' % (socket.gethostname(), port))

    while True:
        try:
            conn, addr = s.accept()
        except socket.error as e:
            if e[1]=='Interrupted system call': continue
            raise

        LOG.info('Connected to %s', addr)

        ibs = new_handler(conn)

        tid = _thread.start_new_thread(ibs.exec_loop,())

        LOG.debug("Thread ID: %d", tid)
