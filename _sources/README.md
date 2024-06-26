# OGC-4 Interface
This repo interfaces with [OGC-4](https://github.com/gwastro/4-ogc) to read in posterior samples for the LVK GW events.

The `PopulationMcZ` class is the main API interface, and is used to get the Mc-z posterior samples and
priors for all the events, and cache them as a matrix of binned Mc-z weights.


```python
from ogc4_interface.population_mcz import PopulationMcZ

population = PopulationMcZ.load()
population.plot_event_mcz_estimates()
population.filter_events(threshold=0.95)
population.plot_weights()
```

## Events
![](docs/_static/event_mcz_estimates.jpeg)


## Weights
![](docs/_static/weights.jpeg)
