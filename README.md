## Deep Dream Experiments
* Try out variations of google deep dream to try come up with interesting results.
* Instead of setting dy = y at the output layer, try something different and compare results etc.

#### How deep dream works
* Different nodes in various layers tend to learn some features. In case of conv layers these are the filters.
* We select a specific layer and then set its gradient to be equal to the output and then start backpropogating. We then perform gradient ascent on the input image so that it becomes more like the current output.
* If we did gradient descent, then the output gradient (the output in this case), would get minimized, so we would end up removing all the details from the image and try to go towards an image with no features.
* But since we are performing ascent and the output gradient is set to the output, we would modify the input image such that more of the output gradient is present. So we would keep trying to increase the features we see currently.
* Suppose we are using one of the initial layers of the network, lets say it detects diagonal lines and circular patches then we would try to add more of these to the input image even if there are slight amounts of them present in the original image. If we use some of the higher layers of the network, then suppose the nodes in that layer detect eyes and paws, then if our input image had slight traces of these, they would start getting amplified so that more and more of them appear.
* If we perform this at the last layer of a classification network (ie. the actual output layer). Suppose the input image resembled lets say a cat with 60% prob and a dog with 40% (2 class classification), then after a few iterations the input image would be modified such that lots of cat features are added everywhere in the image and it just keeps becoming more and more cat like.

##### Models
* Use .t7 models directly
* For .caffemodel models use loadcaffe (ex.NIN)
* A compressed package of most of the famous models, https://drive.google.com/file/d/0B984IDha34FGcmZXZ0RBZ2p5TDQ/view?usp=sharing
(Contains AlexNet,GoogLeNet,GoogLeNetCars,NIN,OxfordFlowers102,PascalVOC2012,PlacesCNN,ResNet,VGG_F,VGG_S)

##### Usage
* `qlua deepdream.lua <source_img> <layer_max> <iterations> <update_rate>`

##### Possible variations
* Instead of amplifying the entire layer, we could also select a single node and then set its output gradient = output and then for the rest set their gradients to 0. In this manner we would be amplifying the features that we want to, so it would generate more predictable images.
* Apply deep dream on multiple layers together and sum up the gradients while moving backwards
* Consider a network which takes large images as inputs to get higher quality images
