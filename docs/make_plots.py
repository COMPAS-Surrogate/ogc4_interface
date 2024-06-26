import os

from ogc4_interface.population_mcz import PopulationMcZ

os.makedirs("docs/_static", exist_ok=True)

population = PopulationMcZ.load()
fig, _ = population.plot_event_mcz_estimates()
fig.savefig(
    "docs/_static/event_mcz_estimates.jpeg", dpi=70, bbox_inches="tight"
)

population.filter_events(threshold=0.95)
fig = population.plot_weights().get_figure()
fig.savefig("docs/_static/weights.jpeg", dpi=300)
