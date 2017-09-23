import java.io.File
import java.net.URL

import org.bytedeco.javacv._
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.helper.opencv_core.{AbstractCvMat, AbstractCvMemStorage, AbstractCvScalar, AbstractIplImage}
import org.bytedeco.javacpp.helper.opencv_imgproc.{cvDrawContours, cvFindContours}
import org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects
import org.bytedeco.javacpp.indexer._
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgproc._
import org.bytedeco.javacpp.opencv_calib3d._
import org.bytedeco.javacpp.opencv_objdetect._


object Demo {
  @throws[Exception]
  def main(args: Array[String]): Unit = {
    var classifierName: String = null
    if (args.length > 0) classifierName = args(0)
    else {
      val url = new URL("https://raw.github.com/Itseez/opencv/2.4.0/data/haarcascades/haarcascade_frontalface_alt.xml")
      val file = Loader.extractResource(url, null, "classifier", ".xml")
      file.deleteOnExit()
      classifierName = file.getAbsolutePath
    }
    // Preload the opencv_objdetect module to work around a known bug.
    Loader.load(classOf[opencv_objdetect])
    // We can "cast" Pointer objects by instantiating a new object of the desired class.
    val classifier = new opencv_objdetect.CvHaarClassifierCascade(cvLoad(classifierName))
    if (classifier.isNull) {
      System.err.println("Error loading classifier file \"" + classifierName + "\".")
      System.exit(1)
    }
    // The available FrameGrabber classes include OpenCVFrameGrabber (opencv_videoio),
    // DC1394FrameGrabber, FlyCaptureFrameGrabber, OpenKinectFrameGrabber, OpenKinect2FrameGrabber,
    // RealSenseFrameGrabber, PS3EyeFrameGrabber, VideoInputFrameGrabber, and FFmpegFrameGrabber.
    val grabber = FrameGrabber.createDefault(0)
    grabber.start()
    // CanvasFrame, FrameGrabber, and FrameRecorder use Frame objects to communicate image data.
    // We need a FrameConverter to interface with other APIs (Android, Java 2D, or OpenCV).
    val converter = new OpenCVFrameConverter.ToIplImage
    // FAQ about IplImage and Mat objects from OpenCV:
    // - For custom raw processing of data, createBuffer() returns an NIO direct
    //   buffer wrapped around the memory pointed by imageData, and under Android we can
    //   also use that Buffer with Bitmap.copyPixelsFromBuffer() and copyPixelsToBuffer().
    // - To get a BufferedImage from an IplImage, or vice versa, we can chain calls to
    //   Java2DFrameConverter and OpenCVFrameConverter, one after the other.
    // - Java2DFrameConverter also has static copy() methods that we can use to transfer
    //   data more directly between BufferedImage and IplImage or Mat via Frame objects.
    var grabbedImage = converter.convert(grabber.grab)
    val width = grabbedImage.width
    val height = grabbedImage.height
    val grayImage = AbstractIplImage.create(width, height, IPL_DEPTH_8U, 1)
    val rotatedImage = grabbedImage.clone
    // Objects allocated with a create*() or clone() factory method are automatically released
    // by the garbage collector, but may still be explicitly released by calling release().
    // You shall NOT call cvReleaseImage(), cvReleaseMemStorage(), etc. on objects allocated this way.
    val storage = AbstractCvMemStorage.create
    // The OpenCVFrameRecorder class simply uses the CvVideoWriter of opencv_videoio,
    // but FFmpegFrameRecorder also exists as a more versatile alternative.
    val recorder = FrameRecorder.createDefault("output.avi", width, height)
    recorder.start()
    // CanvasFrame is a JFrame containing a Canvas component, which is hardware accelerated.
    // It can also switch into full-screen mode when called with a screenNumber.
    // We should also specify the relative monitor/camera response for proper gamma correction.
    val frame = new CanvasFrame("Some Title", CanvasFrame.getDefaultGamma / grabber.getGamma)
    // Let's create some random 3D rotation...
    val randomR = AbstractCvMat.create(3, 3)
    val randomAxis = AbstractCvMat.create(3, 1)
    // We can easily and efficiently access the elements of matrices and images
    // through an Indexer object with the set of get() and put() methods.
    val Ridx = randomR.createIndexer.asInstanceOf[DoubleIndexer]
    val axisIdx = randomAxis.createIndexer.asInstanceOf[DoubleIndexer]
    axisIdx.put(0L, 0d)
//    axisIdx.put(0, (Math.random - 0.5) / 4, (Math.random - 0.5) / 4, (Math.random - 0.5) / 4)
    cvRodrigues2(randomAxis, randomR, null)
    val f = (width + height) / 2.0
    Ridx.put(0, 2, Ridx.get(0, 2) * f)
    Ridx.put(1, 2, Ridx.get(1, 2) * f)
    Ridx.put(2, 0, Ridx.get(2, 0) / f)
    Ridx.put(2, 1, Ridx.get(2, 1) / f)
    System.out.println(Ridx)
    // We can allocate native arrays using constructors taking an integer as argument.
    val hatPoints = new opencv_core.CvPoint(3)
    while ( {
      frame.isVisible && (grabbedImage = converter.convert(grabber.grab)) != null
    }) {
      cvClearMemStorage(storage)
      // Let's try to detect some faces! but we need a grayscale image...
      cvCvtColor(grabbedImage, grayImage, CV_BGR2GRAY)
      val faces = cvHaarDetectObjects(grayImage, classifier, storage, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH)
      val total = faces.total
      var i = 0
      while ( {
        i < total
      }) {
        val r = new opencv_core.CvRect(cvGetSeqElem(faces, i))
        val x = r.x
        val y = r.y
        val w = r.width
        val h = r.height
        cvRectangle(grabbedImage, cvPoint(x, y), cvPoint(x + w, y + h), AbstractCvScalar.RED, 1, CV_AA, 0)
        // To access or pass as argument the elements of a native array, call position() before.
        hatPoints.position(0).x(x - w / 10).y(y - h / 10)
        hatPoints.position(1).x(x + w * 11 / 10).y(y - h / 10)
        hatPoints.position(2).x(x + w / 2).y(y - h / 2)
//        cvFillConvexPoly(grabbedImage, hatPoints.position(0), 3, AbstractCvScalar.GREEN, CV_AA, 0)

        {
          i += 1; i - 1
        }
      }
      // Let's find some contours! but first some thresholding...
      cvThreshold(grayImage, grayImage, 64, 255, CV_THRESH_BINARY)
      // To check if an output argument is null we may call either isNull() or equals(null).
      var contour = new opencv_core.CvSeq(null)
      cvFindContours(grayImage, storage, contour, Loader.sizeof(classOf[opencv_core.CvContour]), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE)
      while ( {
        contour != null && !contour.isNull
      }) {
        if (contour.elem_size > 0) {
          val points = cvApproxPoly(contour, Loader.sizeof(classOf[opencv_core.CvContour]), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contour) * 0.02, 0)
          cvDrawContours(grabbedImage, points, AbstractCvScalar.BLUE, AbstractCvScalar.BLUE, -1, 1, CV_AA)
        }
        contour = contour.h_next
      }
      cvWarpPerspective(grabbedImage, rotatedImage, randomR)
      val rotatedFrame = converter.convert(rotatedImage)
      frame.showImage(rotatedFrame)
      recorder.record(rotatedFrame)
    }
    frame.dispose()
    recorder.stop()
    grabber.stop()
  }
}