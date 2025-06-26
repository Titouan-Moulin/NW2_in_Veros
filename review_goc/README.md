# Supplementary Material

![Setup](https://github.com/user-attachments/assets/bfe4db90-a57a-48ec-81d3-8a43fccf9d7e)


**Figure:** DINO: 
(a) Bathymetry (depth in meters); (b) Temperature restoring; (c) Salinity restoring; (d) Associated surface density; (e) Latitudinal profile of zonal wind stress forcing (in $N·m^{-2}$); (f) Vertical diffusivity (in $cm^2·s^{-1}$)

---

To conduct this experiment, we used the Versatile Ocean Simulator in pure Python: [Veros](https://veros.readthedocs.io/en/latest/) [Hafner et al., 2018], which is based on pyOM2.0 (Python Ocean Model) developed by [Eden et al., 2011].  
We implemented three experiments using a configuration close to DINO (DIabatic NeverWorld2 Ocean) [Kamm et al., 2025].

DINO is based on the NeverWorld2 configuration.  
NeverWorld2 (NW2) [Marques et al., 2022] is a shallow-water model configuration with an idealized geometry comprising:
- A single rectangular basin spanning over two hemispheres
- A re-entrant channel in the Southern Ocean (SO)

It is forced only by surface wind and surface buoyancy with a non-linear equation of state (EOS).  
In DINO, the central ridge present in NW2 has been removed as it introduced bias by separating the basin in two.

---

## Domain Configuration

The domain (see Figure above) is **50° wide** and is bounded by land at **70°N and 70°S**, with a **re-entrant channel** in the Southern Ocean that spans **45°S to 65°S**. 

- A cubic profile of width 2.5° connects the surface to the abyss (nominal depth: **4000 m**).
- A semi-circular ridge:
  - Depth: **2500 m**
  - Radius: **10°**
  - Thickness: **2°**
  - Location: centered on the western opening of the channel
  - Purpose: blocks deep flow through the channel, mimicking the **Drake Passage ridge**

---

## Surface Forcing

### Temperature and Salinity
- Absolute salinity and conservative temperature are extracted from the **IAP dataset** [Cheng et al., 2020, 2024].
- Data are averaged over 10 years (2010–2019).
- **No seasonal variability** is used.

#### Salinity:
- Initial runs using zonal mean surface salinity gave unrealistic bottom water properties.
- To correct this:
  - Salinity was averaged over the first **200 m** before zonal averaging.
  - This helps mimic sea-ice effects by producing saltier water in high latitudes.
- While this increases equatorial salinity and decreases mid-latitude values compared to observations, the result remains consistent with literature profiles [e.g., Munday et al., 2013; Baker et al., 2020].

#### Temperature:
- Surface temperature is kept as the **zonal mean** of the top vertical grid point.
- Both temperature and salinity are smoothed using `Bspline splrep/splev`.

### Wind Stress

- Zonal and **time-invariant** wind stress is applied.
- Defined as a piecewise cubic function interpolating between:
  - Values: `0, 0.2, -0.1, -0.02, -0.1, 0.1, 0` Pa
  - At latitudes: `-70, -45, -15, 0, 15, 45, 70°`
- Each interpolation node has **zero derivative**, ensuring wind stress and its curl vanish at the boundaries.

### Vertical Diffusivity

- Follows a profile from [Bryan, 1979]:
  - Surface: $\kappa_{min} = 0.305 \times 10^{-4} \text{ m}^2/\text{s}$
  - Bottom: $\kappa_{max} = 1.26 \times 10^{-4} \text{ m}^2/\text{s}$

This configuration constitutes the **Reference case** (neverworld2_clim_bis.py)
Two alternative configurations were tested:
1. **Constant diffusivity**: $\kappa = \kappa_{max}$ (neverworld2_clim_mix_max.py)
2. **Blocked re-entrant channel** (neverworld2_noso.py)

## Runs

This three experiments are run on a coarse resolution grid **1°x1°** for **1000 model years**. 


---
## References

- Hafner et al. (2018). *Veros v0.1 – a fast and versatile ocean simulator in pure Python*  
- Eden et al. (2011). *pyOM2.0: A Python Ocean Model*  
- Kamm et al. (2025). *DINO: A Diabatic Model of Pole-to-Pole Ocean Dynamics to Assess Subgrid Parameterizations across Horizontal Scales*  
- Marques et al. (2022). *NeverWorld2: an idealized model hierarchy to investigate ocean mesoscale eddies across resolutions*  
- Cheng et al. (2020). *Improved Estimates of Changes in Upper Ocean Salinity and the Hydrological Cycle*
- Cheng et al. (2024). *APv4 ocean temperature and ocean heat content gridded dataset*  
- Bryan (1979). *A water mass model of the World Ocean*  
- Munday et al. (2013). *ddy Saturation of Equilibrated Circumpolar Currents.*  
- Baker et al. (2020). *Meridional Overturning Circulation in a Multibasin Model. Part I: Dependence on Southern Ocean Buoyancy Forcing.*


