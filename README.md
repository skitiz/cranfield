# Cranfield Search 

### What is this?
This is very simple search implementation for the cranfield dataset.

### What is the cranfield dataset?
Cranfield is a small curated dataset that is very extensively used in the information retrieval experiments.
In the dataset, there are 226 queries (search terms), 1400 documents, and 1837 (evaluations).
The dataset is supposed to be complete in the sense that the documents that should be returned for each known are known.
This makes the evaluation easier. [Click here more details](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/)

### Can I run this search implementation and see the output?
Yes!. Try to run the class `edu.kennesaw.cs.core.EvalSearch` in Java or the file `eval.py` in Python. The expected output looks like as below:
```text
Final ncdg for all queries is {} 0.6301330740878168
```

### What is nDCG score?
[nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) is a very common metric used in search evaluations. 
Higher nDCG score (close to 1.0 ) describes a search system that gives all the relevant results with most relevant ones on the top.

### What should be the goal for this project?

Try to modify the code such that you can increase the nDCG score. 