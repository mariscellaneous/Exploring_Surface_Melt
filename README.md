# Quantifying Melt

This repository combines:
- **SMRT** (Snow Microwave Radiative Transfer Model Framework; Picard et al., 2018)
- **CFM** (Community Firn Model; Stevens et al., 2020) output as input forcing/structure
- **AMSR-2** brightness temperature data (Meier et al., 2018)

 This approach is described in **Dattler et al. (2024)**, which detects melt on the surface of an ice sheet. The core of this code is in folder "Experiments". This repository has with extra Jupyter Notebooks and Python code dedicated to exploring/testing an inversion for liquid water content, as well. 


---

## Requirements

- Python 3.x
- `numpy`
- `scipy`
- `pyproj`
- `smrt`

You will also need access to:
- Community Firn Model (CFM) outputs used by this workflow
- AMSR-2 Unified L3 Daily 12.5 km Brightness Temperatures (Meier et al., 2018)

---

## Data Notes

### AMSR-2
This mockup uses the AMSR-E/AMSR2 Unified L3 Daily 12.5 km product:
- Meier, W. N., Markus, T. & Comiso, J. C. (2018). *AMSR-E/AMSR2 Unified L3 Daily 12.5 km Brightness Temperatures, Sea Ice Concentration, Motion & Snow Depth Polar Grids (AU_SI12, Version 1).* NSIDC DAAC. https://doi.org/10.5067/RA1MIJOYPK3P

### CFM
CFM outputs are used as inputs to the SMRT setup and for comparison to retrieved liquid water content:
- Stevens, C. M., et al. (2020). *The community firn model (cfm) v1.0.* Geoscientific Model Development Discussions, 2020, 1–37.

You may need to edit file paths and variable names in the notebook to match your local CFM/AMSR-2 file structure.

---

## Citation

If you use or adapt this repository, you can consider citing:

- Dattler, M. E., Medley, B., & Stevens, C. M. (2024). *A physics-based Antarctic melt detection technique: combining Advanced Microwave Scanning Radiometer 2, radiative-transfer modeling, and firn modeling.* The Cryosphere, 18(8), 3613–3631.

And the underlying tools/data sources:

- Meier, W. N., Markus, T. & Comiso, J. C. (2018). *AMSR-E/AMSR2 Unified L3 Daily 12.5 km Brightness Temperatures, Sea Ice Concentration, Motion & Snow Depth Polar Grids (AU_SI12, Version 1).* NSIDC DAAC. https://doi.org/10.5067/RA1MIJOYPK3P

- Picard, G., Sandells, M., & Löwe, H. (2018). *SMRT: An active–passive microwave radiative transfer model for snow with multiple microstructure and scattering formulations (v1.0).* Geoscientific Model Development, 11(7), 2763–2788.

- Stevens, C. M., Verjans, V., Lundin, J. M., Kahle, E. C., Horlings, A. N., Horlings, B. I., & Waddington, E. D. (2020). *The community firn model (cfm) v1.0.* Geoscientific Model Development Discussions, 2020, 1–37.

---

## Disclaimer

This repository is a **research mockup** intended for exploration and method prototyping. Results may be sensitive to assumptions, configuration choices, and input data handling. Please validate carefully before drawing scientific conclusions.

---

## Acknowledgements

This work leverages the SMRT model framework (Picard et al., 2018), CFM outputs (Stevens et al., 2020), and AMSR-2 gridded brightness temperature products (Meier et al., 2018), and is informed by the broader methodology developed in Dattler et al. (2024).
