/*
  wavcorr.c
*/

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

inline int min(int x, int y) { return (x < y)? x : y; }
inline int max(int x, int y) { return (x < y)? y : x; }
inline double hann(int i, int n) { return (1.0-cos(2.0*M_PI*i/n))/2.0; }


/* calcsims16: compute the similarity between two vectors. */
double calcsims16(int window, short* seq1, short* seq2)
{
    int i;
    double n1 = 0, n2 = 0, dot = 0;
    for (i = 0; i < window; i++) {
	double x1 = seq1[i]/32768.0;
	double x2 = seq2[i]/32768.0;
	n1 += x1*x1;
	n2 += x2*x2;
	dot += x1*x2;
    }
    return (n1*n2)? (dot/sqrt(n1*n2)) : 0;
}

/* autocorrs16: find the window that has the maximum similarity. */
int autocorrs16(double* psim, int window0, int window1, int length, short* seq)
{
    if (window1 < window0) {
	int x = window1;
	window1 = window0;
	window0 = x;
    }
    /* assert(window0 <= window1); */
  
    int wmax = 0;
    double smax = -1;
    int w;
    for (w = window0; w <= window1; w++) {
	int w1 = window1 - (window1%w);
	if (w1+w <= length) {
	    double s = calcsims16(w1, seq, seq+w);
	    if (smax < s) {
		wmax = w;
		smax = s;
	    }
	}
    }
  
    *psim = smax;
    return wmax;
}

/* autosplices16: find the window that has the maximum similarity. */
int autosplices16(double* psim, int window0, int window1, 
		  int length1, short* seq1, int length2, short* seq2)
{
    if (window1 < window0) {
	int x = window1;
	window1 = window0;
	window0 = x;
    }
    /* assert(window0 <= window1); */
  
    int wmax = 0;
    double smax = -1;
    int w;
    for (w = window0; w <= window1; w++) {
	if (w <= length1 && w <= length2) {
	    double s = calcsims16(w, seq1+length1-w, seq2);
	    if (smax < s) {
		wmax = w;
		smax = s;
	    }
	}
    }
  
    *psim = smax;
    return wmax;
}

/* psolas16: overlap-add two vectors. */
void psolas16(int outlen, short* out, 
	      int length1, short* seq1, 
	      int length2, short* seq2)
{
    int i;

    for (i = 0; i < outlen; i++) {
	/* i < outlen ==> i*length/outlen < length */
	double v = 0;
	if (0 < length1) {
	    /* first half (decreasing) */
	    v += seq1[i*length1/outlen] * hann(i+outlen, outlen*2);
	}
	if (0 < length2) {
	    /* second half (increasing) */
	    v += seq2[i*length2/outlen] * hann(i, outlen*2);
	}
	out[i] = (short)v;
    }
}


/*  Python functions
 */

/* pycalcsims16(window, data1, offset1, data2, offset2); */
static PyObject* pycalcsims16(PyObject* self, PyObject* args)
{
    int window;
    PyObject* data1;
    PyObject* data2;
    int offset1;
    int offset2;

    if (!PyArg_ParseTuple(args, "iOiOi", &window, 
			  &data1, &offset1, &data2, &offset2)) {
	return NULL;
    }

    if (!PyString_CheckExact(data1) ||
	!PyString_CheckExact(data2)) {
	PyErr_SetString(PyExc_TypeError, "Must be string");
	return NULL;
    }

    int length1 = PyString_Size(data1) / sizeof(short);
    int length2 = PyString_Size(data2) / sizeof(short);
    if (window < 0 ||
	offset1 < 0 || length1 < offset1+window ||
	offset2 < 0 || length2 < offset2+window) {
	PyErr_SetString(PyExc_ValueError, "Invalid offset/window");
	return NULL;
    }

    short* seq1 = (short*)PyString_AsString(data1);
    short* seq2 = (short*)PyString_AsString(data2);
    double sim = calcsims16(window, &seq1[offset1], &seq2[offset2]);

    return PyFloat_FromDouble(sim);
}


/* pyautocorrs16(window0, window1, data, offset); */
static PyObject* pyautocorrs16(PyObject* self, PyObject* args)
{
    int window0, window1;
    PyObject* data;
    int offset;

    if (!PyArg_ParseTuple(args, "iiOi", &window0, &window1, &data, &offset)) {
	return NULL;
    }

    if (!PyString_CheckExact(data)) {
	PyErr_SetString(PyExc_TypeError, "Must be string");
	return NULL;
    }

    int length = PyString_Size(data) / sizeof(short);
    if (window0 < 0 || window1 < 0 || 
	offset < 0 || length < offset+window0 || length < offset+window1) {
	PyErr_SetString(PyExc_ValueError, "Invalid offset/window");
	return NULL;
    }

    short* seq = (short*)PyString_AsString(data);
    double smax = 0;
    int wmax = autocorrs16(&smax, window0, window1, length-offset, &seq[offset]);
  
    PyObject* tuple;
    {
	PyObject* v1 = PyInt_FromLong(wmax);
	PyObject* v2 = PyFloat_FromDouble(smax);
	tuple = PyTuple_Pack(2, v1, v2);
	Py_DECREF(v1);
	Py_DECREF(v2);
    }
    return tuple;
}


/* pyautosplices16(window0, window1, data1, data2); */
static PyObject* pyautosplices16(PyObject* self, PyObject* args)
{
    int window0, window1;
    PyObject* data1;
    PyObject* data2;

    if (!PyArg_ParseTuple(args, "iiOO", &window0, &window1, &data1, &data2)) {
	return NULL;
    }

    if (!PyString_CheckExact(data1) ||
	!PyString_CheckExact(data2)) {
	PyErr_SetString(PyExc_TypeError, "Must be string");
	return NULL;
    }

    int length1 = PyString_Size(data1) / sizeof(short);
    int length2 = PyString_Size(data2) / sizeof(short);
    if (window0 < 0 || window1 < 0 ||
	length1 < window0 || length1 < window1 ||
	length2 < window0 || length2 < window1) {
	PyErr_SetString(PyExc_ValueError, "Invalid offset/window");
	return NULL;
    }

    short* seq1 = (short*)PyString_AsString(data1);
    short* seq2 = (short*)PyString_AsString(data2);  
    double smax = 0;
    int wmax = autosplices16(&smax, window0, window1, length1, seq1, length2, seq2);
  
    PyObject* tuple;
    {
	PyObject* v1 = PyInt_FromLong(wmax);
	PyObject* v2 = PyFloat_FromDouble(smax);
	tuple = PyTuple_Pack(2, v1, v2);
	Py_DECREF(v1);
	Py_DECREF(v2);
    }
    return tuple;
}


/* pypsolas16(outlen, 
   offset1, window1, data1,
   offset2, window2, data2); */
static PyObject* pypsolas16(PyObject* self, PyObject* args)
{
    int outlen;
    int offset1, offset2;
    int window1, window2;
    PyObject* data1;
    PyObject* data2;

    if (!PyArg_ParseTuple(args, "iiiOiiO", &outlen,
			  &offset1, &window1, &data1,
			  &offset2, &window2, &data2)) {
	return NULL;
    }
  
    if (!PyString_CheckExact(data1) ||
	!PyString_CheckExact(data2)) {
	PyErr_SetString(PyExc_TypeError, "Must be string");
	return NULL;
    }

    int length1 = PyString_Size(data1) / sizeof(short);
    int length2 = PyString_Size(data2) / sizeof(short);
    if (window1 < 0 || window2 < 0 || 
	offset1 < 0 || length1 < offset1+window1 ||
	offset2 < 0 || length2 < offset2+window2) {
	PyErr_SetString(PyExc_ValueError, "Invalid offset/window");
	return NULL;
    }

    if (outlen <= 0) {
	PyErr_SetString(PyExc_ValueError, "Invalid outlen");
	return NULL;
    }
    short* out = (short*) PyMem_Malloc(sizeof(short)*outlen);
    if (out == NULL) return PyErr_NoMemory();

    short* seq1 = (short*)PyString_AsString(data1);
    short* seq2 = (short*)PyString_AsString(data2);
    psolas16(outlen, out, window1, &seq1[offset1], window2, &seq2[offset2]);
    PyObject* obj = PyString_FromStringAndSize((char*)out, sizeof(short)*outlen);
    PyMem_Free(out);
  
    return obj;
}


/* Module initialization */
PyMODINIT_FUNC
initwavcorr(void)
{
    static PyMethodDef functions[] = {
	{ "calcsims16", (PyCFunction)pycalcsims16, METH_VARARGS,
	  "calcsims16"
	},
	{ "autocorrs16", (PyCFunction)pyautocorrs16, METH_VARARGS,
	  "autocorrs16"
	},
	{ "autosplices16", (PyCFunction)pyautosplices16, METH_VARARGS,
	  "autosplices16"
	},
	{ "psolas16", (PyCFunction)pypsolas16, METH_VARARGS,
	  "psolas16"
	},
	{NULL, NULL},
    };

    Py_InitModule3("wavcorr", functions, "wavcorr"); 
}
