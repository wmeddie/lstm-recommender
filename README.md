# lstm-recommender

An example recommender that uses LSTMs to predict next purchases.  It uses a dataset called OnlineRetail found on
the always amazing __TODO__ University website.  The dataset is representative of the kind of data you will find
on an e-commerce site.


## Training

You can use sbt to train the model.

    sbt "run-main com.yumusoft.lstmrecommender.Train --input trainInput --output output.model --epoch 5"

Example:

    sbt "run-main com.yumusoft.lstmrecommender.Train --input iris.train.csv --output out1.model --epoch 10"

## Evaluation

    sbt "run-main com.yumusoft.lstmrecommender.Evaluate --input testInput --model trained.model"

Example:

    sbt "run-main com.yumusoft.lstmrecommender.Evaluate --input iris.test.csv --model out1.model"

## Repl

Play around with your model with the repl.  It'll import most of DL4J's important classes by default.

    sbt console

