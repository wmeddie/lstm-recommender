package com.yumusoft.lstmrecommender

import java.io.File

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.graph.rnn.{DuplicateToTimeSeriesVertex, LastTimeStepVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, GravesLSTM, OutputLayer, RnnOutputLayer}
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
    .updater(Updater.SGD)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .iterations(1)
    .seed(42)
    .graphBuilder()
    //These are the two inputs to the computation graph
    .addInputs("itemIn")
    .setInputTypes(InputType.recurrent(itemTypeCount))
    .addLayer("encoder", lstm(itemTypeCount, hiddenSize), "itemIn")
    .addVertex("lastTimeStep", new LastTimeStepVertex("itemIn"), "encoder")
    //Create a vertex that allows the duplication of 2d input to a 3d input
    //In this case the last time step of the encoder layer (viz. 2d) is duplicated to the length of the timeseries "sumOut" which is an input to the comp graph
    //Refer to the javadoc for more detail
    //.addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
    //The inputs to the decoder will have size = size of output of last timestep of encoder (numHiddenNodes) + size of the other input to the comp graph,sumOut (feature vector size)
    .addLayer("decoder", lstm(hiddenSize, hiddenSize), "lastTimeStep")
    .addLayer("output", output(hiddenSize, itemTypeCount) , "decoder")
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
    val (trainData, testData) = DataIterators.onlineRetailCsv(c.input)

    log.info("Data Loaded")

    val conf = net(3664, 128)
    val model = new ComputationGraph(conf)
    model.init()

    model.setListeners(new ScoreIterationListener(1))

    for (i <- 0 to c.nEpochs) {
      log.info(s"Starting epoch $i of ${c.nEpochs}")

      while (trainData.hasNext) {
        val ds = trainData.next()
        model.fit(ds)
      }

      log.info(s"Finished epoch $i")
      trainData.reset()
    }

    ModelSerializer.writeModel(model, c.modelName, true)

    log.info(s"Model saved to: ${c.modelName}")
  }
}