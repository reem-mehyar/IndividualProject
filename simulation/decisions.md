# when installing or updating pkg 
python -m pip freeze > requirements-lock.txt

# This document serves as a tracker where I note every assumption I took when recreating the Meng model because it was either not specified in the original paper or was but was modified for suitability. 


Metrics: Num bins not specified for opinion_entropy. 10 bins assumed. Log base also not specified for Shannon entropy. Natural log assumed. 