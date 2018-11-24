## bible.ai

The objective of this script is to crawl through the entire archive of https://www.biblehub.com. By doing such, as can gain content relating to
every single bible version, its verse by verse text, and all of the related cross references.
This is a three-fold project with the objective to discover new relationships between bible verses for pastors, priests, or any spiritual leader.

### pipe.ai - Data Pipeline (<em> Phase 1 of 4 </em>)

Phase 1, collect the data with cross references. This is done by crawling the entire https://www.biblehub.com website. We leverage some structure in the site and predictability with related links, and by such, we construct a large data set with everything we need in our vision.

### transform.ai - Data Transformation (<em> Phase 2 of 4 </em>)

Phase 2, we seek to clean and shape the data. A necessary part to any <em>Machine Learning</em> application. 

### model.ai - Model Training (<em> Phase 3 of 4 </em>)

Phase 3, we seek to utilize the cross references found in the bible as training data. We will learn structure of verses by context and relate that to their cross references. After we do that, we will use a `Recurrent Neural Network (RNN)` to predict based on the sequence of verses without cross references, which ones they should be associated with, to hopefully discover new connections in the bible that were previously not possible to know.

### bible.ai - Model Embedding/Deployment (<em> Phase 4 of 4 </em>)

Phase 4, the goal at this final stage is to have a clean, serialized model that can take any string, from the bible or not, and refer you to exact places in the bible that we believe are highly related to the text you are researching. This can become useful when studying external books, such as `Plato's Republic` or `The Apostolic Fathers` to discover similar verses, that are not explicitely linked to the bible. The goal of this is to augment the users current capability of research with a tool that blends state of the art predictive analysis with real biblical connectivity, previously unseen. 


## Further Research

We want to build the best product for our customers. In this spirit, why stop with the bible? Do you have literature with well known inter-literary references, or a network of references that is `closed-form`? If so, we can expand our work here from just within the bible to accross multiple domains of literature to give you high verse by verse probabilities that the words you're seeking are related. This type of extension makes literary analysis possible between domains such as psychology, social sciences, philosophy, and much more. Please P.M. to discuss details on your custom solution.

This could also be considered as a general application, `lit.ai` to mitigate gaps between social sciences and machine learning.
