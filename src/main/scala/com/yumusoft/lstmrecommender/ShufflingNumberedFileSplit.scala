package com.yumusoft.lstmrecommender

import java.io.{DataInput, DataOutput}
import java.net.URI
import java.nio.file.Paths
import java.util
import java.util.NoSuchElementException

import org.datavec.api.split.InputSplit
import org.slf4j.LoggerFactory

import scala.util.Random

class UriFromPathIterator(paths: util.Iterator[String]) extends util.Iterator[URI] {

  override def hasNext: Boolean = {
    paths.hasNext
  }

  override def next(): URI = {
    if(!hasNext) {
      throw new NoSuchElementException("No next element")
    }
    new URI(paths.next())
  }

  override def remove(): Unit = {
    throw new UnsupportedOperationException()
  }
}

class ShufflingNumberedFileInputSplit(
  seed: Long,
  baseString: String,
  minIdx: Int,
  maxIdx: Int) extends InputSplit {
  private val log = LoggerFactory.getLogger(getClass)
  private val rng = new Random(seed)

  if (baseString == null || !baseString.contains("%d")) {
    throw new IllegalArgumentException("Base String must contain  character sequence %d")
  }

  override def length(): Long = maxIdx - minIdx + 1

  override def locations(): Array[URI] = {
    val uris = rng.shuffle(minIdx to maxIdx).map(i => {
      val path = String.format(baseString, Integer.valueOf(i))

      //log.debug(path)

      Paths.get(path).toUri
    })
    uris.toArray
  }

  def locationsIterator(): util.Iterator[URI] = {
    new UriFromPathIterator(locationsPathIterator())
  }

  def locationsPathIterator(): util.Iterator[String] = {
    new NumberedFileIterator()
  }

  def reset(): Unit = {
    // No op
    // Want next sequences to be random.
  }

  override def write(out: DataOutput): Unit = {
    // No op
  }

  override def readFields(in: DataInput): Unit = {
    // No op
  }

  override def toDouble: Double = {
    throw new UnsupportedOperationException()
  }

  override def toFloat: Float = {
    throw new UnsupportedOperationException()
  }

  override def toInt: Int = {
    throw new UnsupportedOperationException()
  }

  override def toLong: Long = {
    throw new UnsupportedOperationException()
  }

  private class NumberedFileIterator extends util.Iterator[String] {

    private val ids = rng.shuffle(minIdx to maxIdx).toArray
    private var currentId = 0

    override def hasNext: Boolean = currentId <= maxIdx

    override def next(): String = {
      if (!hasNext()) {
        throw new NoSuchElementException()
      }

      val ret = String.format(baseString, Integer.valueOf(ids(currentId)))
      currentId += 1

      ret
    }

    override def remove(): Unit = {
      throw new UnsupportedOperationException()
    }
  }
}