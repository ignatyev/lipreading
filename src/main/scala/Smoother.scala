import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgproc._
import org.bytedeco.javacpp.opencv_imgcodecs._


object Smoother  extends  App{
  def smooth(filename: String): Unit = {
    val image = cvLoadImage(filename)
    if (image != null) {
      cvSmooth(image, image)
      cvSaveImage(filename + "-smooth", image)
      cvReleaseImage(image)
    }
  }

  smooth("C:\\Users\\aignatiev\\Pictures\\Viber\\media-share-0-02-03-b6ee1a3efbccaafd974f624f318d1460525beffbedb0dda9826689b6c7c5b1c6-36b8c434-573e-4b0e-8e6d-1d585c8339b9.jpg")
}