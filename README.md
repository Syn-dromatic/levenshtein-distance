# levenshtein-distance
Compute operational differences between two sequences/texts using the Levenshtein algorithm

___
## Installation:
```
pip install levenshtein-distance
```

___
## Usage:
#### Regular Usage:
```python
from levenshtein_distance import Levenshtein

lev_object = Levenshtein('test', 'text')
distance = lev_object.distance()
ratio = lev_object.ratio()
array = lev_object.sequence_array()
```


#### With replace operation cost of 2:
```python
lev_object = Levenshtein('test', 'text').set_replace_cost(2)
distance = lev_object.distance()
ratio = lev_object.ratio()
array = lev_object.sequence_array()
```
