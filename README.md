### Deep Dream Experiments
* Try out variations of google deep dream to try come up with interesting results.
* Instead of setting dy = y at the output layer, try something different and compare results etc.

##### Models
* Use .t7 models directly
* For .caffemodel models use loadcaffe (ex.NIN)

##### Usage
* `qlua deepdream.lua <source_img> <layer_max> <iterations> <update_rate>`
