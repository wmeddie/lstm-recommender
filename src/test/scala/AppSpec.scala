package com.yumusoft.lstmrecommender

import org.scalatest._
import org.nd4j.linalg.factory.Nd4j

class AppSpec extends FlatSpec with Matchers {

  "This project" should "have some tests" in {
    (2 + 2) should be (4)
  }

  it should "Be able to create NDArrays on the CPU or GPU" in {
    val a = Nd4j.ones(2, 2)
    val b = Nd4j.ones(2, 2)

    (a add b).getDouble(0, 0) should be (2.0)
  }
}