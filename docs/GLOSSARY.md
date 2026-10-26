# TESS Glossary

A reference for the astronomical data, classification systems and engineering terminology used in Astro Data Lab.

## Missions, archives and catalogues

| Acronym | Full name | Meaning |
|---|---|---|
| **TESS** | Transiting Exoplanet Survey Satellite | NASA spacecraft that searches for planets by observing changes in the brightness of stars. |
| **MAST** | Mikulski Archive for Space Telescopes | NASA-supported archive where TESS observations and scientific data products are stored. |
| **TIC** | TESS Input Catalog | Catalogue of astronomical objects, primarily stars. Each object has a unique TIC ID, which we use to link our observations and planetary records. |
| **TOI** | TESS Object of Interest | A potential planetary signal identified for further investigation. One star can have several TOIs. |
| **TCE** | Threshold Crossing Event | A transit-like signal detected by an automated search. Not necessarily a planet or even a vetted candidate. |
| **SPOC** | Science Processing Operations Center | Processing pipeline that produces TESS light curves, pixel files and transit-search results. |
| **QLP** | Quick-Look Pipeline | Alternative pipeline that extracts light curves from TESS full-frame images. |
| **TFOP** | TESS Follow-up Observing Program | Coordinates follow-up observations to investigate candidate planets. |
| **TFOPWG** | TESS Follow-up Observing Program Working Group | Working group responsible for follow-up classifications of TESS candidates. |

## Observation data and terminology

| Acronym | Full name | Meaning |
|---|---|---|
| **FITS** | Flexible Image Transport System | Standard astronomical file format. Our original TESS light curves are stored in FITS files. |
| **LC** | Light Curve | A series of brightness measurements over time, used to identify possible planetary transits. |
| **TPF** | Target Pixel File | Pixel-level observations of a target, allowing us to extract our own light curves. |
| **FFI** | Full-Frame Image | Image covering a complete TESS camera field, rather than just a selected target. |
| **SAP** | Simple Aperture Photometry | Stellar brightness measured by summing light within a selected pixel aperture. |
| **PDCSAP** | Pre-search Data Conditioning Simple Aperture Photometry | SAP brightness measurements corrected for many instrumental and systematic effects. |
| **Sector** | TESS observing sector | A period during which TESS observes a particular region of the sky. Typically around 27 days during the primary mission. |
| **Transit** | Planetary transit | A temporary reduction in stellar brightness caused by a planet passing in front of its host star. |

## Planet classifications

These codes come from the TESS Object of Interest catalogue.

| Code | Classification | Meaning |
|---|---|---|
| **CP** | Confirmed Planet | A planet whose existence has been scientifically confirmed. |
| **KP** | Known Planet | A planet already known from previous observations or surveys. |
| **PC** | Planetary Candidate | A promising planetary signal awaiting confirmation. |
| **APC** | Ambiguous Planetary Candidate | A candidate whose planetary interpretation remains uncertain. |
| **FP** | False Positive | An apparent planetary signal attributed to another astrophysical explanation. |
| **FA** | False Alarm | A detection judged not to be a credible planetary transit signal. |

**Important:** These classifications generally describe individual planet candidates or signals, not entire stars. A star can have multiple planets or candidates with different classifications.

An object absent from these catalogues is **unclassified**, not necessarily planet-free.