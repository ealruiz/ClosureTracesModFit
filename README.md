# ClosureTracesModFit
Python-C++ hybrid implementation for model fitting using closure traces from interferometric visibility data.

**Closure traces** are a new set of **complex closure quantities** (see Broderick & Pesce 2020, Albentosa-Ruiz & Marti-Vidal 2023), with great potential for the exploring the **intrinsic polarization structure** of radio sources, as they encode the information while being **independent of instrumental effects**, thus being robust to calibration effects such as gains and D-terms.

This project aims to use these properties to improve the polarization characterization and imaging in radioastronomy, by computing the $\chi^2$ statistics from the closure traces of different model images to determine the best-fitting model.

## Implementation
For high-performance exploration, we use a Python-C++ hybrid implementation:
1. `_closureTraces.cpp` (C++ Module): provides high-performance functions for efficient numerical computation and multi-threading. Functions include:
  - `setData()`: Initializes the module with the visibility data, such as visibilities UVW coordinates, channel frequencies, times and weights.
  - `loadModel()`: Loads a full-Stokes model image and calculates model visibilities.
  - `getDataTraces()`: Computes closure traces from data visibilities.
  - `getModelTraces()`: Computes closure traces from model visibilities.
  - `getChi2()`: Computes the chi-squared statistic based on visibility data.
2. `closureTraces_method.py`: python wrapper for the C++ module. Defines the `closureTraces` class, which:
    - reads interferometric data from FITS files or CASA Measurement Sets,
    - initializes the C++ engine with the visibility data, and
    - calls the C++ module functions.
3. `setup_closureTraces.py`: build script to compile the C++ module for Python.

## User guide:
To use the scripts, load the visibilities, and perform closure traces-based $\chi^2$ model fitting, follow these steps:

1. **Setup the C++ Module**
Compile the C++ extension by running the following command if working in a **Python environment**:
```sh
python setup_closureTraces.py build_ext --inplace
```
> Or, if working within **CASA**, run:
```sh
/CASAPATH//bin/python3 setup_closureTraces.py build_ext --inplace
```
> **Note:** You need to install the following dependencies before running the setup:
```sh
pip install numpy astropy
```
2. **Integrate the C++ Module** in your Python script.
Import the module and initialize the `closureTraces` class, which loads the visibility data and calls the C++ function `setData()`:
```sh
¡from closureTraces_method import closureTraces
# Initialize closure traces object
CL_TRACES = closureTraces(visname,antennas,cellSize,Npixels,NCPUs)

# Parameters:
# visname  :: Name of the FITS or Measurement Set (MS) file.
# antennas :: List of antenna codes to use. If empty, all antennas are used.
# cellSize :: Pixel size of the model image, formatted as "ValueUnit" (e.g., "1.0muas").
#             Supported units: 'muas', 'mas', 'as', 'arcsec', 'deg', 'rad'.
# Npixels  :: Number of pixels in the model image (assumes a square image).
# NCPUs    :: Number of CPUs to use for multi-core processing.
```

3. **Load Model Images**
You can load the model Images for each Stokes parameters in:
- `CL_TRACES.I[pix_x,pix_y]` $\rightarrow$ Stokes I (total flux density).
- `CL_TRACES.Q[pix_x,pix_y]` & `CL_TRACES.U[pix_x,pix_y]` $\rightarrow$ Linear polarization components.
- `CL_TRACES.V[pix_x,pix_y]` $\rightarrow$ Circular polarization component.
For model fitting, you can define model images based on physical properties such as:
- the spectral index, for the flux density,
- Polarization properties like EVPA, fractional polarization, or Faraday Rotation.
After defining your model, load it into the C++ module:
```sh
CL_TRACES.loadModel(chanlist=[ch])
```
where chanlist is a list of frequency channels to load.
- If the list is empty (`chanlist=[]`), all frequency channels will be loaded using the same model.
- If a model quantity depends on frequency, you can iterate over multiple frequency channels:
```sh
for ch in loadFreqChan:
    # Define your model image for Stokes I, Q, U, V based on source properties
    CL_TRACES.loadModel(chanlist=[ch])
```
4. **Compute the $\chi^2$ statistic for model fitting**
To compare data and model closure traces, compute the $\chi^2$ statistic:
```sh
chi2 = CL_TRACES.getChi2()
```
5. **Iterate to Optimize the Model**
You can **update the model and the $\chi^2$ value** iteratively by repeating **steps 3 and 4**.
* Example: A Markov Chain Monte Carlo (MCMC) approach is implemented in the script `MCMC_cltraces.py`, located in the folder MCMC_Test.
* This script loads simulations of polarized double sources and uses MCMC to find the best model. The folder also includes example measurement sets for testing.
