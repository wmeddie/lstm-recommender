package com.yumusoft.lstmrecommender

import java.io.File

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.graph.rnn.{DuplicateToTimeSeriesVertex, LastTimeStepVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import scopt.OptionParser

case class TrainConfig(
  input: File = null,
  modelName: String = "",
  nEpochs: Int = 1
)

object TrainConfig {
  val parser = new OptionParser[TrainConfig]("Train") {
    head("lstmrecommender Train", "1.0")

    opt[File]('i', "input")
      .required()
      .valueName("<file>")
      .action( (x, c) => c.copy(input = x) )
      .text("The file with training data.")

    opt[Int]('e', "epoch")
      .action( (x, c) => c.copy(nEpochs = x) )
      .text("Number of times to go over whole training set.")

    opt[String]('o', "output")
      .required()
      .valueName("<modelName>")
      .action( (x, c) => c.copy(modelName = x) )
      .text("Name of trained model file.")
  }

  def parse(args: Array[String]): Option[TrainConfig] = {
    parser.parse(args, TrainConfig())
  }
}

object Train {
  private val log = LoggerFactory.getLogger(getClass)

  private def embedding(in: Int, out: Int): EmbeddingLayer =
    new EmbeddingLayer.Builder()
      .nIn(in)
      .nOut(out)
      .build()

  private def lstm(nIn: Int, size: Int): GravesLSTM =
    new GravesLSTM.Builder()
      .nIn(nIn)
      .nOut(size)
      .activation(Activation.SOFTSIGN)
      .build()

  private def output(nIn: Int, nOut: Int): RnnOutputLayer =
    new RnnOutputLayer.Builder()
      .nIn(nIn)
      .nOut(nOut)
      .activation(Activation.SOFTMAX)
      .lossFunction(LossFunctions.LossFunction.MCXENT)
      .build()

  private def net(itemTypeCount: Int, hiddenSize: Int) = new NeuralNetConfiguration.Builder()
    .weightInit(WeightInit.XAVIER)
    .learningRate(0.001)
    .updater(Updater.RMSPROP)
    .rmsDecay(0.95)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .iterations(10)
    .seed(42)
    .graphBuilder()
    //These are the two inputs to the computation graph
    .addInputs("itemIn")
    .setInputTypes(InputType.recurrent(itemTypeCount))
    .addLayer("embed", embedding(itemTypeCount, hiddenSize), "itemIn")
    .addLayer("lstm1", lstm(hiddenSize, hiddenSize), "embed")
    .addLayer("output", output(hiddenSize, itemTypeCount) , "lstm1")
    .setOutputs("output")
    .build()

  def main(args: Array[String]): Unit = {
    TrainConfig.parse(args) match {
      case Some(config) =>
        log.info("Starting training")

        train(config)

        log.info("Training finished.")
      case _ =>
        log.error("Invalid arguments.")
    }
  }

  private def train(c: TrainConfig): Unit = {
    val trainData = DataIterators.onlineRetailCsv(c.input)

    log.info("Data Loaded")

    val conf = net(trainData.inputColumns(), 128)
    val model = new ComputationGraph(conf)
    model.init()

    model.setListeners(new ScoreIterationListener(1))

    for (i <- 0 to c.nEpochs) {
      log.info(s"Starting epoch $i of ${c.nEpochs}")

      while (trainData.hasNext) {
        val ds = trainData.next()
        log.info(ds.getFeatures.toString)
        log.info("===============")
        model.fit(ds)
      }

      log.info(s"Finished epoch $i")
      trainData.reset()
    }

    ModelSerializer.writeModel(model, c.modelName, true)

    log.info(s"Model saved to: ${c.modelName}")
  }
}