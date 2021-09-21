/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissAssert.h>
#include <faiss/python/python_callbacks.h>

namespace {

struct PyThreadLock {
    PyGILState_STATE gstate;

    PyThreadLock() {
        gstate = PyGILState_Ensure();
    }

    ~PyThreadLock() {
        PyGILState_Release(gstate);
    }
};

}; // namespace

/***********************************************************
 * Callbacks for IO reader and writer
 ***********************************************************/

PyCallbackIOWriter::PyCallbackIOWriter(PyObject* callback, size_t bs)
        : callback(callback), bs(bs) {
    PyThreadLock gil;
    Py_INCREF(callback);
    name = "PyCallbackIOWriter";
}

size_t PyCallbackIOWriter::operator()(
        const void* ptrv,
        size_t size,
        size_t nitems) {
    size_t ws = size * nitems;
    const char* ptr = (const char*)ptrv;
    PyThreadLock gil;
    while (ws > 0) {
        size_t wi = ws > bs ? bs : ws;
        PyObject* result = PyObject_CallFunction(
                callback, "(N)", PyBytes_FromStringAndSize(ptr, wi));
        if (result == NULL) {
            FAISS_THROW_MSG("py err");
        }
        // TODO check nb of bytes written
        ptr += wi;
        ws -= wi;
        Py_DECREF(result);
    }
    return nitems;
}

PyCallbackIOWriter::~PyCallbackIOWriter() {
    PyThreadLock gil;
    Py_DECREF(callback);
}

PyCallbackIOReader::PyCallbackIOReader(PyObject* callback, size_t bs)
        : callback(callback), bs(bs) {
    PyThreadLock gil;
    Py_INCREF(callback);
    name = "PyCallbackIOReader";
}

size_t PyCallbackIOReader::operator()(void* ptrv, size_t size, size_t nitems) {
    size_t rs = size * nitems;
    size_t nb = 0;
    char* ptr = (char*)ptrv;
    PyThreadLock gil;
    while (rs > 0) {
        size_t ri = rs > bs ? bs : rs;
        PyObject* result = PyObject_CallFunction(callback, "(n)", ri);
        if (result == NULL) {
            FAISS_THROW_MSG("propagate py error");
        }
        if (!PyBytes_Check(result)) {
            Py_DECREF(result);
            FAISS_THROW_MSG("read callback did not return a bytes object");
        }
        size_t sz = PyBytes_Size(result);
        if (sz == 0) {
            Py_DECREF(result);
            break;
        }
        nb += sz;
        if (sz > rs) {
            Py_DECREF(result);
            FAISS_THROW_FMT(
                    "read callback returned %zd bytes (asked %zd)", sz, rs);
        }
        memcpy(ptr, PyBytes_AsString(result), sz);
        Py_DECREF(result);
        ptr += sz;
        rs -= sz;
    }
    return nb / size;
}

PyCallbackIOReader::~PyCallbackIOReader() {
    PyThreadLock gil;
    Py_DECREF(callback);
}
