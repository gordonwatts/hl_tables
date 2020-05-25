# hl_tables

 A high level tables dispatcher for putting together multiple tables executors

## Examples

## Making a histogram

```python
dataset = EventDataset(f'localds://mc16_13TeV:{ds["RucioDSName"].values[0]}')
df = xaod_table(dataset)
truth = df.TruthParticles('TruthParticles')
llp_truth = truth[truth.pdgId == 35]
histogram(llp_truth.Count(), bins=3, range=(0,3))
plt.yscale('log')
plt.xlabel('Number of good LLPs in each event')
plt.ylabel('a MC Sample')
```

1. The histogram data will be calculated by the backend and returned to your local Jupyter instance.
1. Plots will be rendered!

## Outstanding things

- Definitely need to decide on an approach to this whole thing. Reducers - and where should they be applied, at the outer most or inner most level? So seq.count() - should that mean seq.Select(a: a.count()), or seq.count() (number of events, or a list of objects inside the event)?

- Count needs to be changed to num or dimensions, etc.
