#include <pthread.h>
#include <Python.h>

#if PY_MAJOR_VERSION >= 3
#define NPY_NO_DEPRECATED_API 0x0
#endif
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <new>
#include <ctime>
#include <sys/stat.h>
#include <string.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <vector>

#define EPSILON 0.00001

bool use32BitsVariables = true; // true -> 32 bits; false -> 64 bits

// cribbed from SWIG machinery
#if PY_MAJOR_VERSION >= 3
#define PyClass_Check(obj) PyObject_IsInstance(obj, (PyObject *)&PyType_Type)
#define PyInt_Check(x) PyLong_Check(x)
#define PyInt_AsLong(x) PyLong_AsLong(x)
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define PyInt_FromSize_t(x) PyLong_FromSize_t(x)
#define PyString_Check(name) PyBytes_Check(name)
#define PyString_FromString(x) PyUnicode_FromString(x)
#define PyString_Format(fmt, args)  PyUnicode_Format(fmt, args)
#define PyString_Size(str) PyBytes_Size(str)
#define PyString_InternFromString(key) PyUnicode_InternFromString(key)
#define Py_TPFLAGS_HAVE_CLASS Py_TPFLAGS_BASETYPE
#define PyString_AS_STRING(x) PyUnicode_AS_STRING(x)
#define _PyLong_FromSsize_t(x) PyLong_FromSsize_t(x)
#endif

// and after some hacking
#if PY_MAJOR_VERSION >= 3
#define PyString_AsString(obj) PyUnicode_AsUTF8(obj)
#endif

/* Docstrings */
static char module_docstring[] =
        "Compute closure traces and compare them to models.";

static char setData_docstring[] =
        "Sets the pointers to data and metadata.";

static char loadModel_docstring[] =
        "Sets the pointers to Image Model.";

static char getDataTraces_docstring[] =
        "Computes the data traces of a given quadruplet.";

static char getModelTraces_docstring[] =
        "Computes the data traces of a given quadruplet.";

static char getChi2_docstring[] =
        "Computes the Chi2 for all traces. The model has to be set first.";


/* Available functions */
static PyObject *setData(PyObject *self, PyObject *args);
static PyObject *loadModel(PyObject *self, PyObject *args);
static PyObject *getDataTraces(PyObject *self, PyObject *args);
static PyObject *getModelTraces(PyObject *self, PyObject *args);
static PyObject *getChi2(PyObject *self, PyObject *args);


/* Module specification */
static PyMethodDef module_methods[] = {
        {"setData", setData, METH_VARARGS, setData_docstring},
        {"loadModel", loadModel, METH_VARARGS, loadModel_docstring},
        {"getDataTraces", getDataTraces, METH_VARARGS, getDataTraces_docstring},
        {"getModelTraces", getModelTraces, METH_VARARGS, getModelTraces_docstring},
        {"getChi2", getChi2, METH_VARARGS, getChi2_docstring},
        {NULL, NULL, 0, NULL}   /* terminated by list of NULLs, apparently */
};


// Trick to allow type promotion below
template <typename T>
struct identity_t { typedef T type; };

/// Make working with std::complex<> nubmers suck less... allow promotion.
#define COMPLEX_OPS(OP)                                                 \
  template <typename _Tp>                                               \
  std::complex<_Tp>                                                     \
  operator OP(std::complex<_Tp> lhs, const typename identity_t<_Tp>::type & rhs) \
  {                                                                     \
    return lhs OP rhs;                                                  \
  }                                                                     \
  template <typename _Tp>                                               \
  std::complex<_Tp>                                                     \
  operator OP(const typename identity_t<_Tp>::type & lhs, const std::complex<_Tp> & rhs) \
  {                                                                     \
    return lhs OP rhs;                                                  \
  }
COMPLEX_OPS(+)
COMPLEX_OPS(-)
COMPLEX_OPS(*)
COMPLEX_OPS(/)
#undef COMPLEX_OPS

////////////////////
// GLOBAL VARIABLES HERE:
#define USE_32BIT_VARIABLES 1 // Set to 0 for 64-bit types

#if USE_32BIT_VARIABLES
typedef std::complex<float> cplxNbits;
typedef float floatNbits;
typedef std::complex<float> visibility;
typedef float weight;
#else
typedef std::complex<double> cplxNbits;
typedef double floatNbits;
typedef std::complex<double> visibility;
typedef double weight;
#endif


int nQuads, nChan, nTimes, nBas;

visibility ImagU = cplxNbits(0.,1.);

visibility **DATA, **MODEL;
visibility *output;
weight **WEIGHTS;
weight *output_WEIGHT;

double *Times;

floatNbits **UVW, *TwoPiFreq;
floatNbits *RA, *DEC, *MODEL_I, *MODEL_Q, *MODEL_U, *MODEL_V;
std::vector<int> loadChan;

bool **OBSERVED;
int **QUADRUPLETS;

int nPix, nPixSq;
bool *isRAzero, *isDECzero;

bool isRL;

double *ChiSq;
double *ChiSqTraces;

bool save_chi2_distribution;
std::string fname = "chi2_distribution.dat";

int NCPU;
struct WORKER {int thread; int t0; int t1;};


/* Initialize the module */

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef pc_module_def = {
        PyModuleDef_HEAD_INIT,
        "_closureTraces",       /* m_name */
        module_docstring,       /* m_doc */
        -1,                     /* m_size */
        module_methods,         /* m_methods */
        NULL,NULL,NULL,NULL     /* m_reload, m_traverse, m_clear, m_free */
};
PyMODINIT_FUNC PyInit__closureTraces(void)
{
    PyObject *m = PyModule_Create(&pc_module_def);
    import_array();
    return(m);
}
#else
PyMODINIT_FUNC init_XPCal(void)
{
    import_array();
    PyObject *m = Py_InitModule3("_closureTraces", module_methods, module_docstring);
    if (m == NULL)
        return;
}
#endif




//////////////////////////////////
// MAIN FUNCTION:
static PyObject *setData(PyObject *self, PyObject *args)
{
    int i,auxRL,saveChisq;
    
    // Object to return:
    PyObject *ret;
    
    ret = Py_BuildValue("i",-1);
    
    // Function arguments:
    PyObject *PyData, *PyWeights, *PyModel, *PyObserved, *PyQuadruplets, *PyOutput, *PyTimes, *PyUVW, *PyTwoPiFreq, *PyOutputWeight;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOiii",
            &PyTimes, &PyUVW, &PyTwoPiFreq,
            &PyData, &PyWeights, &PyModel, &PyObserved, &PyQuadruplets,
            &PyOutput, &PyOutputWeight,
            &NCPU, &auxRL, &saveChisq)){
        printf("FAILED setData! Wrong arguments!\n");
        fflush(stdout);
        return ret;
    };
    isRL   = auxRL==1;
    save_chi2_distribution = saveChisq==1;
    
    ChiSq  = new double[NCPU];
    ChiSqTraces = new double[NCPU];
    
    nQuads = PyList_Size(PyQuadruplets);
    nBas   = PyList_Size(PyData);
    nTimes = (int) PyArray_DIM(PyList_GetItem(PyData,0),0);
    nChan  = (int) PyArray_DIM(PyList_GetItem(PyData,0),1);
    printf("\n\n FROM C++:   %i  %i  %i  %i\n\n",nQuads, nBas, nTimes, nChan);
    Times     = (double *) PyArray_DATA(PyTimes);
    TwoPiFreq = (floatNbits *) PyArray_DATA(PyTwoPiFreq);
    UVW       = new floatNbits*[nBas];
    DATA      = new visibility*[nBas];
    MODEL     = new visibility*[nBas];
    OBSERVED  = new bool*[nBas];
    QUADRUPLETS = new int*[nQuads];
    WEIGHTS   = new weight*[nBas];
    
    for (i=0; i<nBas;i++){
        UVW[i]   = (floatNbits *)PyArray_DATA(PyList_GetItem(PyUVW,i));
        DATA[i]  = (visibility *)PyArray_DATA(PyList_GetItem(PyData,i));
        MODEL[i] = (visibility *)PyArray_DATA(PyList_GetItem(PyModel,i));
        OBSERVED[i] = (bool *)PyArray_DATA(PyList_GetItem(PyObserved,i));
        WEIGHTS[i] = (weight *)PyArray_DATA(PyList_GetItem(PyWeights,i));
    };
    
    for (i=0; i<nQuads;i++){
        QUADRUPLETS[i] = (int *)PyArray_DATA(PyList_GetItem(PyQuadruplets,i));
    };
    
    output = (visibility *)PyArray_DATA(PyOutput);
    output_WEIGHT = (weight *)PyArray_DATA(PyOutputWeight);
    
    //int ti = 5, chi = 3, stki = 1;
    //printf("\n\n FIRST VISIBILITY C++:   %.3e  %.3e | %i\n\n",DATA[2][ti*nChan*4 + chi*4 +stki].real(),DATA[2][ti*nChan*4 + chi*4 +stki].imag(),OBSERVED[2][ti]);
    
    ret = Py_BuildValue("i",0);
    return ret;
}
//////////////////////////////////


//////////////////////////////////
// Parallel version to compute model visibilities:
void *computeUVModel(void *work){
    
    WORKER *Iam = (WORKER *)work;
    
    int t0 = Iam->t0;
    int t1 = Iam->t1;
    int thread = Iam->thread;
    
    int i,j,k,p1y,p1x,pcol,ptot,bl;
    int tJump,nuJump,tJumpPol,uvJump;
    floatNbits vDec, uRA; 
    floatNbits *cosDec1, *sinDec1, *cosRA1, *sinRA1;
    visibility *myVISIB, *myMODEL;
    weight *WGT;
    floatNbits *myUVW;
    
    cosDec1 = new floatNbits[nPix];
    sinDec1 = new floatNbits[nPix];
    cosRA1  = new floatNbits[nPix];
    sinRA1  = new floatNbits[nPix];
    
    floatNbits cosP, sinP; 
    visibility Phasor;
    ChiSq[thread] = 0.0;
    
    for(bl=0; bl<nBas; bl++){
        for(j=t0; j<t1; j++){
            if(OBSERVED[bl][j]){
                tJump = j*nChan; tJumpPol = tJump*4; uvJump = j*3;
                
                myUVW   = &UVW[bl][uvJump];
                myVISIB = &DATA[bl][tJumpPol];
                myMODEL = &MODEL[bl][tJumpPol];
                WGT     = &WEIGHTS[bl][tJump];
                
                // Loop over channels:
                for(int ch : loadChan){
                    nuJump = ch*4;
                    myMODEL[nuJump]   = 0.;
                    myMODEL[nuJump+1] = 0.;
                    myMODEL[nuJump+2] = 0.;
                    myMODEL[nuJump+3] = 0.;
                    // Now, the sines and cosines are only computed once, for each RA and Dec (in radians):
                    for(p1y=0;p1y<nPix;p1y++){
                        // UVW in uvfits: time-light units, i.e. (UU',VV') = (UU,VV)/c -> (UU',VV')*freq = (UU,VV)/lambda
                        // 2*pi*(UU,VV)*(RA,dec)/lambda = 2*pi*(UU',VV')*freq*(RA,dec)
                        if(!isRAzero[p1y]){
                            uRA = RA[p1y]*myUVW[0]*TwoPiFreq[ch];
                            cosRA1[p1y] = std::cos(uRA);
                            sinRA1[p1y] = std::sin(uRA);
                        } else {
                            cosRA1[p1y] = 0.0; sinRA1[p1y] = 0.0;
                        };
                        if(!isDECzero[p1y]){
                            vDec = DEC[p1y]*myUVW[1]*TwoPiFreq[ch];
                            cosDec1[p1y] = std::cos(vDec);
                            sinDec1[p1y] = std::sin(vDec);
                        } else {
                            cosDec1[p1y] = 0.0; sinDec1[p1y] = 0.0;
                        };
                    };
                    // Loop over all pixels:
                    for(p1y=0;p1y<nPix;p1y++){
                        if(!isRAzero[p1y]){
                            pcol = p1y*nPix;
                            for(p1x=0;p1x<nPix;p1x++){
                                ptot = pcol + p1x;
                                if(MODEL_I[ptot]!=0.0){
                                    cosP   = cosRA1[p1y]*cosDec1[p1x] - sinRA1[p1y]*sinDec1[p1x];
                                    sinP   = cosRA1[p1y]*sinDec1[p1x] + sinRA1[p1y]*cosDec1[p1x];
                                    Phasor = visibility(cosP,sinP);
                                    //if(thread==0 && bl == 1){printf("%i %i %i %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e \n", bl, p1x, p1y,TwoPiFreq[i], uRA, vDec, RA[p1y], DEC[p1x], myUVW[0], myUVW[1], Phasor.real(), Phasor.imag());};
                                    if(isRL){
                                        myMODEL[nuJump]   += (MODEL_I[ptot] + MODEL_V[ptot])*Phasor; // RR
                                        myMODEL[nuJump+1] += (MODEL_I[ptot] - MODEL_V[ptot])*Phasor; // LL
                                        myMODEL[nuJump+2] += (MODEL_Q[ptot] + MODEL_U[ptot]*ImagU)*Phasor; // RL
                                        myMODEL[nuJump+3] += (MODEL_Q[ptot] - MODEL_U[ptot]*ImagU)*Phasor; // LR
                                    } else {
                                        myMODEL[nuJump]   += (MODEL_I[ptot] + MODEL_Q[ptot])*Phasor; // XX
                                        myMODEL[nuJump+1] += (MODEL_I[ptot] - MODEL_Q[ptot])*Phasor; // YY
                                        myMODEL[nuJump+2] += (MODEL_U[ptot] + MODEL_V[ptot]*ImagU)*Phasor; // XY
                                        myMODEL[nuJump+3] += (MODEL_U[ptot] - MODEL_V[ptot]*ImagU)*Phasor; // YX
                                    };
                                };
                            };
                        };
                    };
                    for(k=0;k<4;k++){
                        Phasor = myMODEL[nuJump+k]-myVISIB[nuJump+k];
                        ChiSq[thread] += WGT[ch]*(Phasor.real()*Phasor.real() + Phasor.imag()*Phasor.imag());
                    };
                };
            };
        };
    };
    
    delete[] cosRA1;
    delete[] cosDec1;
    delete[] sinRA1;
    delete[] sinDec1;
    
    pthread_exit((void*) 0);
}

// Function to load the Model. Calls computeUVModel (parallelization) 
static PyObject *loadModel(PyObject *self, PyObject *args)
{
    PyObject *Err;
    void *status;
    int i,j,k,pcol;
    
    PyObject *RAPy, *DECPy;
    PyObject *MODEL_IPy, *MODEL_QPy, *MODEL_UPy, *MODEL_VPy, *CHAN_Py;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &RAPy, &DECPy,
            &MODEL_IPy, &MODEL_QPy, &MODEL_UPy, &MODEL_VPy, &CHAN_Py)){
        printf("FAILED loadModel! Unable to parse arguments!\n");
        fflush(stdout);
        Err = Py_BuildValue("i",-1);
        return Err;
    };
    
// List of channels where the model will be loaded
    int nchan  = PyArray_SIZE(CHAN_Py);
    if (nchan == 0) {
        // If no specific channels are selected, range from 0 to nChan
        loadChan.clear();
        for (int i = 0; i < nChan; ++i){loadChan.push_back(i);};
    } else {
        // Populate loadChan with the data from CHAN_Py
        int *loadChanPy;
        loadChanPy = (int *)PyArray_DATA(CHAN_Py);
        loadChan.assign(loadChanPy, loadChanPy + nchan);
        //printf("\nLoading %i channels:",nchan);
        //for (int ch : loadChan){printf(" %i,",ch);};
        //printf("\n");
    }
    
// Memory to store the model image:
    nPix = (int) PyArray_DIM(RAPy,0);
    nPixSq = nPix*nPix;
    
    isRAzero = new bool[nPix];
    isDECzero = new bool[nPix];
    
    RA = (floatNbits *)PyArray_DATA(RAPy);
    DEC = (floatNbits *)PyArray_DATA(DECPy);
    MODEL_I = (floatNbits *)PyArray_DATA(MODEL_IPy);
    MODEL_Q = (floatNbits *)PyArray_DATA(MODEL_QPy);
    MODEL_U = (floatNbits *)PyArray_DATA(MODEL_UPy);
    MODEL_V = (floatNbits *)PyArray_DATA(MODEL_VPy);
    
    // Figure out if all pixels in a row/column are zero
    for(j=0;j<nPix;j++){
        isRAzero[j] = true;
        isDECzero[j] = true;
        pcol = j*nPix;
        for(k=0;k<nPix;k++){
            if(MODEL_I[pcol+k] != 0.0){isRAzero[j]=false; break;};
        };
        for(k=0;k<nPix;k++){
            if(MODEL_I[j+k*nPix] != 0.0){isDECzero[j]=false; break;};
        };
    };
    
    // Define threads:
    pthread_t MyThreads[NCPU];
    pthread_attr_t attr;
    
    // Information for the workers:
    WORKER *workers = new WORKER[NCPU];
    
    ///////////////////
    // Distribute works:
    int timesPerCPU = nTimes/NCPU;
    int timesRemainder = nTimes%NCPU;
    
    int currRem = 0;
    int currT = 0;
    for(i=0; i<NCPU; i++){
        workers[i].t0 = currT;
        workers[i].t1 = currT + timesPerCPU;
        workers[i].thread = i;
        if(currRem < timesRemainder){
            workers[i].t1 += 1;
        };
        currRem += 1;
        currT = workers[i].t1;
    };
    ///////////////////

    // Execute work:
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    for (i=0; i<NCPU; i++){
        pthread_create(&MyThreads[i], &attr, computeUVModel, (void *)&workers[i]);
    };
    
    pthread_attr_destroy(&attr);
    
    // Join threads and compute total ChiSq:
    double TotChiSq = 0.0;
    for(i=0; i<NCPU; i++){
        pthread_join(MyThreads[i], &status);
        TotChiSq += ChiSq[i];
    };
    
    // Clean memory:
    delete[] workers;
    delete[] isRAzero;
    delete[] isDECzero;
    
    //printf("\n Model loaded. ChiSq: %.6e\n",TotChiSq);
    
    Err = Py_BuildValue("f",TotChiSq);
    return Err;
}
//////////////////////////////////

//////////////////////////////////
// Function to get the ClosureTraces from the data, for a given quadruplet
static PyObject *getDataTraces(PyObject *self, PyObject *args)
{
    // Object to return:
    PyObject *ret;

    ret = Py_BuildValue("i",-1);
    int quad;

    if (!PyArg_ParseTuple(args, "i", &quad)){printf("FAILED getDataTraces! Wrong arguments!\n"); fflush(stdout);  return ret;};

    // printf("Will process baselines %i %i %i %i\n",QUADRUPLETS[quad][0],QUADRUPLETS[quad][1],QUADRUPLETS[quad][2],QUADRUPLETS[quad][3]);

    int *q = QUADRUPLETS[quad];
    int t, nu, i;
    bool isThere;

    visibility *V01, *V02, *V03, *V12, *V13, *V23;
    weight *W01, *W02, *W03, *W12, *W13, *W23;

    visibility RRab, LLab, RLab, LRab;
    visibility RRbc, LLbc, RLbc, LRbc;
    visibility RRad, LLad, RLad, LRad;
    visibility RRcd, LLcd, RLcd, LRcd;

    int tJump,nuJump,tJumpPol;

    for (t=0;t<nTimes;t++){
        isThere = true;
        // TODO TODO Improve logic to not ignore all quadruplets (e.g., intrasites missing)
        for (i=0;i<6;i++){
            if(!OBSERVED[q[i]][t]){isThere=false;break;};
        };
        // STOKES ORDER (FROM PYTHON): RR LL RL LR
        if(isThere){
            tJump = t*nChan; tJumpPol = tJump*4;

            V01 = &DATA[q[0]][tJumpPol];
            V02 = &DATA[q[1]][tJumpPol];
            V03 = &DATA[q[2]][tJumpPol];
            V12 = &DATA[q[3]][tJumpPol];
            V13 = &DATA[q[4]][tJumpPol];
            V23 = &DATA[q[5]][tJumpPol];

            W01 = &WEIGHTS[q[0]][tJump];
            W02 = &WEIGHTS[q[1]][tJump];
            W03 = &WEIGHTS[q[2]][tJump];
            W12 = &WEIGHTS[q[3]][tJump];
            W13 = &WEIGHTS[q[4]][tJump];
            W23 = &WEIGHTS[q[5]][tJump];

            for(nu=0;nu<nChan;nu++){
                nuJump = nu*4;
                RRab = V01[nuJump]; LLab = V01[nuJump+1]; RLab = V01[nuJump+2]; LRab = V01[nuJump+3];
                RRbc = std::conj(V12[nuJump]); LLbc = std::conj(V12[nuJump+1]); RLbc = std::conj(V12[nuJump+2]); LRbc = std::conj(V12[nuJump+3]);
                RRad = V03[nuJump]; LLad = V03[nuJump+1]; RLad = V03[nuJump+2]; LRad = V03[nuJump+3];
                RRcd = V23[nuJump]; LLcd = V23[nuJump+1]; RLcd = V23[nuJump+2]; LRcd = V23[nuJump+3];

                output[tJump+nu] = 0.5*(-((LRbc*RRab - RLab*RRbc)*LRcd + (RLab*RLbc - LLbc*RRab)*RRcd)*LLad + ((LRbc*RRab - RLab*RRbc)*LLcd + (RLab*RLbc - LLbc*RRab)*RLcd)*LRad + ((LRab*LRbc - LLab*RRbc)*LRcd - (LLbc*LRab - LLab*RLbc)*RRcd)*RLad - ((LRab*LRbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*RLbc)*RLcd)*RRad)/((LRad*RLad - LLad*RRad)*(LRbc*RLbc - LLbc*RRbc));
                output_WEIGHT[tJump+nu] = std::pow(W01[nu]*W03[nu]*W12[nu]*W23[nu],1.0/4.0);
            };
        } else {
            tJump = t*nChan;
            for (nu=0;nu<nChan;nu++){output[tJump+nu] = 0.0;output_WEIGHT[tJump+nu] = 0.0;};
        };
    };

    ret = Py_BuildValue("i",0);
    return ret;
}

// Function to get the Cl.Traces from the model, for a given quadruplet
static PyObject *getModelTraces(PyObject *self, PyObject *args)
{
    // Object to return:
    PyObject *ret;

    ret = Py_BuildValue("i",-1);
    int quad;

    if (!PyArg_ParseTuple(args, "i", &quad)){printf("FAILED getModelTraces! Wrong arguments!\n"); fflush(stdout);  return ret;};

    // printf("Will process baselines %i %i %i %i\n",QUADRUPLETS[quad][0],QUADRUPLETS[quad][1],QUADRUPLETS[quad][2],QUADRUPLETS[quad][3]);

    int *q = QUADRUPLETS[quad];
    int t, nu, i;
    bool isThere;

    visibility *V01, *V02, *V03, *V12, *V13, *V23;
    weight *W01, *W02, *W03, *W12, *W13, *W23;

    visibility RRab, LLab, RLab, LRab;
    visibility RRbc, LLbc, RLbc, LRbc;
    visibility RRad, LLad, RLad, LRad;
    visibility RRcd, LLcd, RLcd, LRcd;

    int tJump, nuJump,tJumpPol;

    for (t=0;t<nTimes;t++){
        isThere = true;
        for (i=0;i<6;i++){
            if(!OBSERVED[q[i]][t]){isThere=false;break;};
        };
        // STOKES ORDER (FROM PYTHON): RR LL RL LR
        if(isThere){
            tJump = t*nChan; tJumpPol = tJump*4;

            V01 = &MODEL[q[0]][tJumpPol];
            V02 = &MODEL[q[1]][tJumpPol];
            V03 = &MODEL[q[2]][tJumpPol];
            V12 = &MODEL[q[3]][tJumpPol];
            V13 = &MODEL[q[4]][tJumpPol];
            V23 = &MODEL[q[5]][tJumpPol];

            W01 = &WEIGHTS[q[0]][tJump];
            W02 = &WEIGHTS[q[1]][tJump];
            W03 = &WEIGHTS[q[2]][tJump];
            W12 = &WEIGHTS[q[3]][tJump];
            W13 = &WEIGHTS[q[4]][tJump];
            W23 = &WEIGHTS[q[5]][tJump];

            for(nu=0;nu<nChan;nu++){
                nuJump = nu*4;
                RRab = V01[nuJump]; LLab = V01[nuJump+1]; RLab = V01[nuJump+2]; LRab = V01[nuJump+3];
                RRbc = std::conj(V12[nuJump]); LLbc = std::conj(V12[nuJump+1]); RLbc = std::conj(V12[nuJump+2]); LRbc = std::conj(V12[nuJump+3]);
                RRad = V03[nuJump]; LLad = V03[nuJump+1]; RLad = V03[nuJump+2]; LRad = V03[nuJump+3];
                RRcd = V23[nuJump]; LLcd = V23[nuJump+1]; RLcd = V23[nuJump+2]; LRcd = V23[nuJump+3];

                output[tJump+nu] = 0.5*(-((LRbc*RRab - RLab*RRbc)*LRcd + (RLab*RLbc - LLbc*RRab)*RRcd)*LLad + ((LRbc*RRab - RLab*RRbc)*LLcd + (RLab*RLbc - LLbc*RRab)*RLcd)*LRad + ((LRab*LRbc - LLab*RRbc)*LRcd - (LLbc*LRab - LLab*RLbc)*RRcd)*RLad - ((LRab*LRbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*RLbc)*RLcd)*RRad)/((LRad*RLad - LLad*RRad)*(LRbc*RLbc - LLbc*RRbc));
                output_WEIGHT[tJump+nu] = std::pow(W01[nu]*W03[nu]*W12[nu]*W23[nu],1.0/4.0);
            };
        } else {
            tJump = t*nChan;
            for (nu=0;nu<nChan;nu++){output[tJump+nu] = 0.0;output_WEIGHT[tJump+nu] = 0.0;};
        };
    };

    ret = Py_BuildValue("i",0);
    return ret;
}
//////////////////////////////////

//////////////////////////////////
// Parallel version to get the chi2 from the data vs model cl.Traces:
void *computeChiSq(void *work){
    
    WORKER *Iam = (WORKER *)work;
    
    int t0 = Iam->t0;
    int t1 = Iam->t1;
    int thread = Iam->thread;
    
    ChiSqTraces[thread] = 0.0;
    
    std::vector<double> ChiSqRe;
    std::vector<double> ChiSqIm;
    
    int *qid;
    int q, t, nu, i;
    bool isThere, isWgt;
    
    visibility *V01, *V02, *V03, *V12, *V13, *V23;
    visibility *M01, *M02, *M03, *M12, *M13, *M23;
    
    visibility RRab, LLab, RLab, LRab;
    visibility RRbc, LLbc, RLbc, LRbc;
    visibility RRad, LLad, RLad, LRad;
    visibility RRcd, LLcd, RLcd, LRcd;
    visibility DATUM, MOD;
    
    int tJump, nuJump,tJumpPol;
    
    for(q=0; q<nQuads; q++){
        qid = QUADRUPLETS[q];
        for (t=t0; t<t1; t++){
            isThere = true;
            
            for (i=0;i<6;i++){
                if(!OBSERVED[qid[i]][t]){isThere=false;break;};
            };
            // if(OBSERVED[qid[0]][t] && OBSERVED[qid[3]][t] && OBSERVED[qid[5]][t] && OBSERVED[qid[2]][t]){isThere=true;}else{isThere=false;};
            
            // STOKES ORDER (FROM PYTHON): RR LL RL LR
            if(isThere){
                tJump = t*nChan; tJumpPol = tJump*4;
                
                V01 = &DATA[qid[0]][tJumpPol];
                V02 = &DATA[qid[1]][tJumpPol];
                V03 = &DATA[qid[2]][tJumpPol];
                V12 = &DATA[qid[3]][tJumpPol];
                V13 = &DATA[qid[4]][tJumpPol];
                V23 = &DATA[qid[5]][tJumpPol];
                
                M01 = &MODEL[qid[0]][tJumpPol];
                M02 = &MODEL[qid[1]][tJumpPol];
                M03 = &MODEL[qid[2]][tJumpPol];
                M12 = &MODEL[qid[3]][tJumpPol];
                M13 = &MODEL[qid[4]][tJumpPol];
                M23 = &MODEL[qid[5]][tJumpPol];
                
                for(nu=0;nu<nChan;nu++){
                    isWgt = true;
                    for(i=0;i<6;i++){if(WEIGHTS[qid[i]][tJump+nu]==0.0){isWgt=false;break;};};
                    // if(WEIGHTS[qid[0]][tJump+nu] && WEIGHTS[qid[3]][tJump+nu] && WEIGHTS[qid[5]][tJump+nu] && WEIGHTS[qid[2]][tJump+nu]){isWgt=true;}else{isWgt=false;};
                    
                    if(isWgt){
                        nuJump = nu*4;
                        // cl trace 0123
                        RRab = V01[nuJump]; LLab = V01[nuJump+1]; RLab = V01[nuJump+2]; LRab = V01[nuJump+3];
                        RRbc = std::conj(V12[nuJump]); LLbc = std::conj(V12[nuJump+1]); RLbc = std::conj(V12[nuJump+2]); LRbc = std::conj(V12[nuJump+3]);
                        RRad = V03[nuJump]; LLad = V03[nuJump+1]; RLad = V03[nuJump+2]; LRad = V03[nuJump+3];
                        RRcd = V23[nuJump]; LLcd = V23[nuJump+1]; RLcd = V23[nuJump+2]; LRcd = V23[nuJump+3];
                        
                        DATUM = 0.5*(-((LRbc*RRab - RLab*RRbc)*LRcd + (RLab*RLbc - LLbc*RRab)*RRcd)*LLad + ((LRbc*RRab - RLab*RRbc)*LLcd + (RLab*RLbc - LLbc*RRab)*RLcd)*LRad + ((LRab*LRbc - LLab*RRbc)*LRcd - (LLbc*LRab - LLab*RLbc)*RRcd)*RLad - ((LRab*LRbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*RLbc)*RLcd)*RRad)/((LRad*RLad - LLad*RRad)*(LRbc*RLbc - LLbc*RRbc));
                        
                        RRab = M01[nuJump]; LLab = M01[nuJump+1]; RLab = M01[nuJump+2]; LRab = M01[nuJump+3];
                        RRbc = std::conj(M12[nuJump]); LLbc = std::conj(M12[nuJump+1]); RLbc = std::conj(M12[nuJump+2]); LRbc = std::conj(M12[nuJump+3]);
                        RRad = M03[nuJump]; LLad = M03[nuJump+1]; RLad = M03[nuJump+2]; LRad = M03[nuJump+3];
                        RRcd = M23[nuJump]; LLcd = M23[nuJump+1]; RLcd = M23[nuJump+2]; LRcd = M23[nuJump+3];
                        
                        MOD = 0.5*(-((LRbc*RRab - RLab*RRbc)*LRcd + (RLab*RLbc - LLbc*RRab)*RRcd)*LLad + ((LRbc*RRab - RLab*RRbc)*LLcd + (RLab*RLbc - LLbc*RRab)*RLcd)*LRad + ((LRab*LRbc - LLab*RRbc)*LRcd - (LLbc*LRab - LLab*RLbc)*RRcd)*RLad - ((LRab*LRbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*RLbc)*RLcd)*RRad)/((LRad*RLad - LLad*RRad)*(LRbc*RLbc - LLbc*RRbc));
                        /*
                        if(t<7){
                            printf("SPW%i 0-1-2-3 Time: %.7e\n",nu,(Times[t]/86400.-57849.)*24.);
                            printf("MODEL RRab: %.5e + %.5e j LLab: %.5e + %.5e j RLab: %.5e + %.5e j LRab: %.5e + %.5e j\n",RRab.real(),RRab.imag(),LLab.real(),LLab.imag(),RLab.real(),RLab.imag(),LRab.real(),LRab.imag());
                            printf("MODEL RRbc: %.5e + %.5e j LLbc: %.5e + %.5e j RLbc: %.5e + %.5e j LRbc: %.5e + %.5e j\n",RRbc.real(),RRbc.imag(),LLbc.real(),LLbc.imag(),LRbc.real(),LRbc.imag(),RLbc.real(),RLbc.imag());
                            printf("MODEL RRcd: %.5e + %.5e j LLcd: %.5e + %.5e j RLcd: %.5e + %.5e j LRcd: %.5e + %.5e j\n",RRcd.real(),RRcd.imag(),LLcd.real(),LLcd.imag(),RLcd.real(),RLcd.imag(),LRcd.real(),LRcd.imag());
                            printf("MODEL RRad: %.5e + %.5e j LLad: %.5e + %.5e j RLad: %.5e + %.5e j LRad: %.5e + %.5e j\n",RRad.real(),RRad.imag(),LLad.real(),LLad.imag(),RLad.real(),RLad.imag(),LRad.real(),LRad.imag());

                            printf("SPW%i 0-1-2-3 Time: %.7e TraceMod: %.7e + %.7e j TraceDat: %.7e + %.7e j\n",nu,(Times[t]/86400.-57849.)*24.,MOD.real(),MOD.imag(),DATUM.real(),DATUM.imag());
                        };
								*/
                        
                        MOD = DATUM-MOD;
                        ChiSqTraces[thread] += MOD.real()*MOD.real() + MOD.imag()*MOD.imag();
                        if (save_chi2_distribution){ ChiSqRe.push_back(MOD.real()); ChiSqIm.push_back(MOD.imag()); };
                        
                        // cl trace 0132
                        RRab = V01[nuJump]; LLab = V01[nuJump+1]; RLab = V01[nuJump+2]; LRab = V01[nuJump+3];
                        RRbc = std::conj(V13[nuJump]); LLbc = std::conj(V13[nuJump+1]); RLbc = std::conj(V13[nuJump+2]); LRbc = std::conj(V13[nuJump+3]);
                        RRad = V02[nuJump]; LLad = V02[nuJump+1]; RLad = V02[nuJump+2]; LRad = V02[nuJump+3];
                        RRcd = std::conj(V23[nuJump]); LLcd = std::conj(V23[nuJump+1]); RLcd = std::conj(V23[nuJump+2]); LRcd = std::conj(V23[nuJump+3]);
                        
                        DATUM = 0.5*(-((LRbc*RRab - RLab*RRbc)*RLcd + (RLab*RLbc - LLbc*RRab)*RRcd)*LLad + ((LRbc*RRab - RLab*RRbc)*LLcd + (RLab*RLbc - LLbc*RRab)*LRcd)*LRad + ((LRab*LRbc - LLab*RRbc)*RLcd - (LLbc*LRab - LLab*RLbc)*RRcd)*RLad - ((LRab*LRbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*RLbc)*LRcd)*RRad)/((LRad*RLad - LLad*RRad)*(LRbc*RLbc - LLbc*RRbc));
                        
                        RRab = M01[nuJump]; LLab = M01[nuJump+1]; RLab = M01[nuJump+2]; LRab = M01[nuJump+3];
                        RRbc = std::conj(M13[nuJump]); LLbc = std::conj(M13[nuJump+1]); RLbc = std::conj(M13[nuJump+2]); LRbc = std::conj(M13[nuJump+3]);
                        RRad = M02[nuJump]; LLad = M02[nuJump+1]; RLad = M02[nuJump+2]; LRad = M02[nuJump+3];
                        RRcd = std::conj(M23[nuJump]); LLcd = std::conj(M23[nuJump+1]); RLcd = std::conj(M23[nuJump+2]); LRcd = std::conj(M23[nuJump+3]);
                        
                        MOD = 0.5*(-((LRbc*RRab - RLab*RRbc)*RLcd + (RLab*RLbc - LLbc*RRab)*RRcd)*LLad + ((LRbc*RRab - RLab*RRbc)*LLcd + (RLab*RLbc - LLbc*RRab)*LRcd)*LRad + ((LRab*LRbc - LLab*RRbc)*RLcd - (LLbc*LRab - LLab*RLbc)*RRcd)*RLad - ((LRab*LRbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*RLbc)*LRcd)*RRad)/((LRad*RLad - LLad*RRad)*(LRbc*RLbc - LLbc*RRbc));
                        /*
                        if(t<7){
                            printf("SPW%i 0-1-3-2 Time: %.7e TraceMod: %.5e + %.5e j TraceDat: %.5e + %.5e j\n",nu,(Times[t]/86400.-57849.)*24.,MOD.real(),MOD.imag(),DATUM.real(),DATUM.imag());
                        };
								*/
                        
                        MOD = DATUM-MOD;
                        ChiSqTraces[thread] += MOD.real()*MOD.real() + MOD.imag()*MOD.imag();
                        if (save_chi2_distribution){ ChiSqRe.push_back(MOD.real()); ChiSqIm.push_back(MOD.imag()); };
                        
                        // CL TRACE 0213
                        RRab = V02[nuJump]; LLab = V02[nuJump+1]; RLab = V02[nuJump+2]; LRab = V02[nuJump+3];
                        RRbc = V12[nuJump]; LLbc = V12[nuJump+1]; RLbc = V12[nuJump+2]; LRbc = V12[nuJump+3];
                        RRad = V03[nuJump]; LLad = V03[nuJump+1]; RLad = V03[nuJump+2]; LRad = V03[nuJump+3];
                        RRcd = V13[nuJump]; LLcd = V13[nuJump+1]; RLcd = V13[nuJump+2]; LRcd = V13[nuJump+3];
                        
                        DATUM = 0.5*(-((RLbc*RRab - RLab*RRbc)*LRcd + (RLab*LRbc - LLbc*RRab)*RRcd)*LLad + ((RLbc*RRab - RLab*RRbc)*LLcd + (RLab*LRbc - LLbc*RRab)*RLcd)*LRad + ((LRab*RLbc - LLab*RRbc)*LRcd - (LLbc*LRab - LLab*LRbc)*RRcd)*RLad - ((LRab*RLbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*LRbc)*RLcd)*RRad)/((LRad*RLad - LLad*RRad)*(RLbc*LRbc - LLbc*RRbc));
                        
                        RRab = M02[nuJump]; LLab = M02[nuJump+1]; RLab = M02[nuJump+2]; LRab = M02[nuJump+3];
                        RRbc = M12[nuJump]; LLbc = M12[nuJump+1]; RLbc = M12[nuJump+2]; LRbc = M12[nuJump+3];
                        RRad = M03[nuJump]; LLad = M03[nuJump+1]; RLad = M03[nuJump+2]; LRad = M03[nuJump+3];
                        RRcd = M13[nuJump]; LLcd = M13[nuJump+1]; RLcd = M13[nuJump+2]; LRcd = M13[nuJump+3];
                        
                        MOD = 0.5*(-((RLbc*RRab - RLab*RRbc)*LRcd + (RLab*LRbc - LLbc*RRab)*RRcd)*LLad + ((RLbc*RRab - RLab*RRbc)*LLcd + (RLab*LRbc - LLbc*RRab)*RLcd)*LRad + ((LRab*RLbc - LLab*RRbc)*LRcd - (LLbc*LRab - LLab*LRbc)*RRcd)*RLad - ((LRab*RLbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*LRbc)*RLcd)*RRad)/((LRad*RLad - LLad*RRad)*(RLbc*LRbc - LLbc*RRbc));
                        /*
                        if(t<7){
                            printf("SPW%i 0-2-1-3 Time: %.7e TraceMod: %.5e + %.5e j TraceDat: %.5e + %.5e j\n",nu,(Times[t]/86400.-57849.)*24.,MOD.real(),MOD.imag(),DATUM.real(),DATUM.imag());
                        };
								*/
                        
                        MOD = DATUM-MOD;
                        ChiSqTraces[thread] += MOD.real()*MOD.real() + MOD.imag()*MOD.imag();
                        if (save_chi2_distribution){ ChiSqRe.push_back(MOD.real()); ChiSqIm.push_back(MOD.imag()); };
                        
                        // CL TRACE 0231
                        RRab = V02[nuJump]; LLab = V02[nuJump+1]; RLab = V02[nuJump+2]; LRab = V02[nuJump+3];
                        RRbc = std::conj(V23[nuJump]); LLbc = std::conj(V23[nuJump+1]); RLbc = std::conj(V23[nuJump+2]); LRbc = std::conj(V23[nuJump+3]);
                        RRad = V01[nuJump]; LLad = V01[nuJump+1]; RLad = V01[nuJump+2]; LRad = V01[nuJump+3];
                        RRcd = std::conj(V13[nuJump]); LLcd = std::conj(V13[nuJump+1]); RLcd = std::conj(V13[nuJump+2]); LRcd = std::conj(V13[nuJump+3]);
                        
                        DATUM = 0.5*(-((LRbc*RRab - RLab*RRbc)*RLcd + (RLab*RLbc - LLbc*RRab)*RRcd)*LLad + ((LRbc*RRab - RLab*RRbc)*LLcd + (RLab*RLbc - LLbc*RRab)*LRcd)*LRad + ((LRab*LRbc - LLab*RRbc)*RLcd - (LLbc*LRab - LLab*RLbc)*RRcd)*RLad - ((LRab*LRbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*RLbc)*LRcd)*RRad)/((LRad*RLad - LLad*RRad)*(LRbc*RLbc - LLbc*RRbc));
                        
                        RRab = M02[nuJump]; LLab = M02[nuJump+1]; RLab = M02[nuJump+2]; LRab = M02[nuJump+3];
                        RRbc = std::conj(M23[nuJump]); LLbc = std::conj(M23[nuJump+1]); RLbc = std::conj(M23[nuJump+2]); LRbc = std::conj(M23[nuJump+3]);
                        RRad = M01[nuJump]; LLad = M01[nuJump+1]; RLad = M01[nuJump+2]; LRad = M01[nuJump+3];
                        RRcd = std::conj(M13[nuJump]); LLcd = std::conj(M13[nuJump+1]); RLcd = std::conj(M13[nuJump+2]); LRcd = std::conj(M13[nuJump+3]);
                        
                        MOD = 0.5*(-((LRbc*RRab - RLab*RRbc)*RLcd + (RLab*RLbc - LLbc*RRab)*RRcd)*LLad + ((LRbc*RRab - RLab*RRbc)*LLcd + (RLab*RLbc - LLbc*RRab)*LRcd)*LRad + ((LRab*LRbc - LLab*RRbc)*RLcd - (LLbc*LRab - LLab*RLbc)*RRcd)*RLad - ((LRab*LRbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*RLbc)*LRcd)*RRad)/((LRad*RLad - LLad*RRad)*(LRbc*RLbc - LLbc*RRbc));
                        /*
                        if(t<7){
                            printf("SPW%i 0-2-3-1 Time: %.7e TraceMod: %.5e + %.5e j TraceDat: %.5e + %.5e j\n",nu,(Times[t]/86400.-57849.)*24.,MOD.real(),MOD.imag(),DATUM.real(),DATUM.imag());
                        };
								*/
                        
                        MOD = DATUM-MOD;
                        ChiSqTraces[thread] += MOD.real()*MOD.real() + MOD.imag()*MOD.imag();
                        if (save_chi2_distribution){ ChiSqRe.push_back(MOD.real()); ChiSqIm.push_back(MOD.imag()); };
                        
                        // cl trace 0312
                        RRab = V03[nuJump]; LLab = V03[nuJump+1]; RLab = V03[nuJump+2]; LRab = V03[nuJump+3];
                        RRbc = V13[nuJump]; LLbc = V13[nuJump+1]; RLbc = V13[nuJump+2]; LRbc = V13[nuJump+3];
                        RRad = V02[nuJump]; LLad = V02[nuJump+1]; RLad = V02[nuJump+2]; LRad = V02[nuJump+3];
                        RRcd = V12[nuJump]; LLcd = V12[nuJump+1]; RLcd = V12[nuJump+2]; LRcd = V12[nuJump+3];
                        
                        DATUM = 0.5*(-((RLbc*RRab - RLab*RRbc)*LRcd + (RLab*LRbc - LLbc*RRab)*RRcd)*LLad + ((RLbc*RRab - RLab*RRbc)*LLcd + (RLab*LRbc - LLbc*RRab)*RLcd)*LRad + ((LRab*RLbc - LLab*RRbc)*LRcd - (LLbc*LRab - LLab*LRbc)*RRcd)*RLad - ((LRab*RLbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*LRbc)*RLcd)*RRad)/((LRad*RLad - LLad*RRad)*(RLbc*LRbc - LLbc*RRbc));
                        
                        RRab = M03[nuJump]; LLab = M03[nuJump+1]; RLab = M03[nuJump+2]; LRab = M03[nuJump+3];
                        RRbc = M13[nuJump]; LLbc = M13[nuJump+1]; RLbc = M13[nuJump+2]; LRbc = M13[nuJump+3];
                        RRad = M02[nuJump]; LLad = M02[nuJump+1]; RLad = M02[nuJump+2]; LRad = M02[nuJump+3];
                        RRcd = M12[nuJump]; LLcd = M12[nuJump+1]; RLcd = M12[nuJump+2]; LRcd = M12[nuJump+3];
                        
                        MOD = 0.5*(-((RLbc*RRab - RLab*RRbc)*LRcd + (RLab*LRbc - LLbc*RRab)*RRcd)*LLad + ((RLbc*RRab - RLab*RRbc)*LLcd + (RLab*LRbc - LLbc*RRab)*RLcd)*LRad + ((LRab*RLbc - LLab*RRbc)*LRcd - (LLbc*LRab - LLab*LRbc)*RRcd)*RLad - ((LRab*RLbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*LRbc)*RLcd)*RRad)/((LRad*RLad - LLad*RRad)*(RLbc*LRbc - LLbc*RRbc));
                        /*
                        if(t<7){
                            printf("SPW%i 0-3-1-2 Time: %.7e TraceMod: %.5e + %.5e j TraceDat: %.5e + %.5e j\n",nu,(Times[t]/86400.-57849.)*24.,MOD.real(),MOD.imag(),DATUM.real(),DATUM.imag());
                        };
								*/
                        
                        MOD = DATUM-MOD;
                        ChiSqTraces[thread] += MOD.real()*MOD.real() + MOD.imag()*MOD.imag();
                        if (save_chi2_distribution){ ChiSqRe.push_back(MOD.real()); ChiSqIm.push_back(MOD.imag()); };
                        
                        // cl trace 0321
                        RRab = V03[nuJump]; LLab = V03[nuJump+1]; RLab = V03[nuJump+2]; LRab = V03[nuJump+3];
                        RRbc = V23[nuJump]; LLbc = V23[nuJump+1]; RLbc = V23[nuJump+2]; LRbc = V23[nuJump+3];
                        RRad = V01[nuJump]; LLad = V01[nuJump+1]; RLad = V01[nuJump+2]; LRad = V01[nuJump+3];
                        RRcd = std::conj(V12[nuJump]); LLcd = std::conj(V12[nuJump+1]); RLcd = std::conj(V12[nuJump+2]); LRcd = std::conj(V12[nuJump+3]);
                        
                        DATUM = 0.5*(-((RLbc*RRab - RLab*RRbc)*RLcd + (RLab*LRbc - LLbc*RRab)*RRcd)*LLad + ((RLbc*RRab - RLab*RRbc)*LLcd + (RLab*LRbc - LLbc*RRab)*LRcd)*LRad + ((LRab*RLbc - LLab*RRbc)*RLcd - (LLbc*LRab - LLab*LRbc)*RRcd)*RLad - ((LRab*RLbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*LRbc)*LRcd)*RRad)/((LRad*RLad - LLad*RRad)*(RLbc*LRbc - LLbc*RRbc));
                        
                        RRab = M03[nuJump]; LLab = M03[nuJump+1]; RLab = M03[nuJump+2]; LRab = M03[nuJump+3];
                        RRbc = M23[nuJump]; LLbc = M23[nuJump+1]; RLbc = M23[nuJump+2]; LRbc = M23[nuJump+3];
                        RRad = M01[nuJump]; LLad = M01[nuJump+1]; RLad = M01[nuJump+2]; LRad = M01[nuJump+3];
                        RRcd = std::conj(M12[nuJump]); LLcd = std::conj(M12[nuJump+1]); RLcd = std::conj(M12[nuJump+2]); LRcd = std::conj(M12[nuJump+3]);
                        
                        MOD = 0.5*(-((RLbc*RRab - RLab*RRbc)*RLcd + (RLab*LRbc - LLbc*RRab)*RRcd)*LLad + ((RLbc*RRab - RLab*RRbc)*LLcd + (RLab*LRbc - LLbc*RRab)*LRcd)*LRad + ((LRab*RLbc - LLab*RRbc)*RLcd - (LLbc*LRab - LLab*LRbc)*RRcd)*RLad - ((LRab*RLbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*LRbc)*LRcd)*RRad)/((LRad*RLad - LLad*RRad)*(RLbc*LRbc - LLbc*RRbc));
                        /*
                        if(t<7){
                            printf("SPW%i 0-3-2-1 Time: %.7e TraceMod: %.5e + %.5e j TraceDat: %.5e + %.5e j\n",nu,(Times[t]/86400.-57849.)*24.,MOD.real(),MOD.imag(),DATUM.real(),DATUM.imag());
                        };
								*/
                        
                        MOD = DATUM-MOD;
                        ChiSqTraces[thread] += MOD.real()*MOD.real() + MOD.imag()*MOD.imag();
                        if (save_chi2_distribution){ ChiSqRe.push_back(MOD.real()); ChiSqIm.push_back(MOD.imag()); };
                        
                        /*
                        if(t%100<=0.0005 or t<7){
                            printf("SPW%i chi2 t%i: %.5f\n",nu,t,ChiSqTraces[thread]);
                        };
								*/                        
                    };
                };
            };
        };
    };
    
    if (save_chi2_distribution)
    {
        int NchiSq = ChiSqRe.size();
        std::ofstream outfile(fname);
        for (int cindx = 0; cindx < NchiSq; ++cindx)
        {
            outfile << ChiSqRe[cindx] << "\t" << ChiSqIm[cindx] << "\n";
        };
        outfile.close();
    };
    
    pthread_exit((void*) 0);
}

// Function to get the chi2 from the data vs model cl.Traces
static PyObject *getChi2(PyObject *self, PyObject *args)
{
    PyObject *PyChi2;
    void *status;
    int i;
    
    // Define threads:
    pthread_t MyThreads[NCPU];
    pthread_attr_t attr;
    
    // Information for the workers:
    WORKER *workers = new WORKER[NCPU];
    
    ///////////////////
    // Distribute works:
    int timesPerCPU = nTimes/NCPU;
    int timesRemainder = nTimes%NCPU;
    
    int currRem = 0;
    int currT = 0;
    for(i=0; i<NCPU; i++){
        workers[i].t0 = currT;
        workers[i].t1 = currT + timesPerCPU;
        workers[i].thread = i;
        if(currRem < timesRemainder){
            workers[i].t1 += 1;
        };
        currRem += 1;
        currT = workers[i].t1;
    };
    ///////////////////
    
    // Execute work:
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    for (i=0; i<NCPU; i++){
        pthread_create(&MyThreads[i], &attr, computeChiSq, (void *)&workers[i]);
    };
    
    pthread_attr_destroy(&attr);
    
    // Join threads and compute total ChiSq:
    double TotChiSq = 0.0;
    for(i=0; i<NCPU; i++){
        pthread_join(MyThreads[i], &status);
        TotChiSq += ChiSqTraces[i];
    };
    
    // Clean memory:
    delete[] workers;
    
    //printf("\n Model Closure Traces computed. ChiSq: %.6e\n",TotChiSq);
    
    PyChi2 = Py_BuildValue("d",TotChiSq);
    return PyChi2;
}
//////////////////////////////////



