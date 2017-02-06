# lstm-recommender

An example recommender that uses LSTMs to predict next purchases.  It uses a dataset called OnlineRetail found on
the always amazing UCI Machine Learning repository website.  The dataset is representative of the kind of data you will find
on an e-commerce site.


## Training

You can use sbt to train the model.

    sbt ""run-main com.yumusoft.lstmrecommender.Train --input retail-small --output model.out --epoch 100 --hidden 10 --count 10""


## Repl

Play around with your model with the repl.  It'll import most of DL4J's important classes by default.

    sbt console

