# Meeting script: five-minute overview

**Recovering ENSO-related structure from sea-surface temperature**

> Speaking draft for the Neogi meeting. Timing is approximate and includes figure pauses. Stage directions are not spoken. Before presenting, add one accurate sentence identifying your specific contribution: which implementation, analysis, or interpretation you personally completed. Repository contents alone do not establish individual ownership. Do not use the placeholder as a claim.

## Question

[Show an SST anomaly map.]

The question in this project is whether a few coordinates can summarize ENSO-related variation in monthly sea-surface temperature, and whether a nonlinear method adds anything useful beyond PCA.

ENSO is El Niño–Southern Oscillation. Its warm and cool phases affect the tropical Pacific. Each monthly temperature map contains thousands of grid-cell values, so the computational task is to turn that map into a small number of coordinates without using an ENSO index as the fitting target.

[Insert your verified contribution sentence here.]

## Data and workflow

The recorded analysis uses NOAA sea-surface temperature from 1950 through 2024. It covers the tropical Pacific, with 900 months and about 2,300 retained grid cells.

The workflow subtracts the average seasonal cycle, so the input describes departures from the usual temperature for each calendar month. It then fits two representations to those monthly fields and compares their coordinates with Niño 3.4, a regional average of SST anomalies.

That reference shares the physical basis of the input. It helps identify an ENSO-related coordinate, but it is not an independent measurement of a different phenomenon. The reference is also used to choose the sign of the plotted coordinates, since an eigenvector can be multiplied by minus one without changing its meaning.

## Linear baseline

[Show the leading EOF and PC1 versus Niño 3.4.]

PCA finds the spatial pattern that accounts for the most variation. The pattern is called an EOF in climate science, and its strength in each month is a principal component.

The saved results report a correlation of 0.946 between the first principal component and Niño 3.4. That tells us the leading variance direction is closely related to this familiar SST index. These are existing reported results; I am not presenting them as newly reproduced measurements.

## Nonlinear comparison

[Show the PCA and diffusion-map comparison.]

Diffusion maps starts from a different idea. It connects months that have similar temperature patterns, then uses eigenvectors of that similarity graph to assign coordinates. This lets it make a nonlinear representation of the data.

At the normalization setting used for the main comparison, the first nonconstant diffusion coordinate has a reported correlation of 0.923 with Niño 3.4. PCA is higher on this metric. So the result is not that the nonlinear method outperforms the linear one.

The diffusion-map plot provides another way to inspect warm and cool states and selected trajectories. But a curved plot is not proof of a physical attractor. Some of that shape can come from the coordinate transformation itself. Establishing a physical meaning for the additional coordinates would take more work.

## Interpretation and next step

There are two important limits. First, this is a same-month comparison. The representation and seasonal reference are estimated from the full sample. It is not a forecast, and it is not a held-out test.

Second, the original significance check shuffled individual months of the index. Climate time series have persistence, and that shuffle destroys it. I would not use the original sigma claims as valid evidence of time-series significance.

A useful next step would be a temporal holdout, with preprocessing and model fitting restricted to the training period, plus a null test that respects temporal dependence. Those are proposed checks, not changes already implemented.

The main takeaway is modest: both methods recover ENSO-related variation in the recorded sample, and PCA is already a strong baseline. The nonlinear view is interesting, but it needs a clearer evaluation before claiming added value. The repository also includes delay embeddings, wavelets, and a linear evolution model. I would keep those as backup rather than make them the center of this comparison.
