import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class Main {

    private static final String IMG = System.getProperty("user.dir") + "\\img\\test.png";
    private static final String IMGPATH = System.getProperty("user.dir") + "\\img\\";
    private static final String OPENCVPATH = System.getProperty("user.dir") + "\\opencv\\";

    public static void main(String[] args) {
        imgCorrecting();
    }

    public static void faceDetector() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        CascadeClassifier classifier = new CascadeClassifier(OPENCVPATH + "cascade.xml");
        Mat img = Imgcodecs.imread(IMG);
        MatOfRect ofRect = new MatOfRect();
        classifier.detectMultiScale(img, ofRect);

        System.out.println(ofRect.toArray().length);

        for (Rect rect : ofRect.toArray()) {
            Imgproc.rectangle(img,
                    new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }
        Imgcodecs.imwrite(IMGPATH + "faceDetect.jpg", img);
    }

    /**
     * 图片校正
     */
    public static void imgCorrecting() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat mat = Imgcodecs.imread(IMG);
        Mat grayMat = new Mat();

        // 灰度
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY);

        // 二值化 - 黑白
        Mat binaryMat = new Mat(grayMat.height(), grayMat.width(), CvType.CV_8UC1);
        Imgproc.threshold(grayMat, binaryMat, 110, 255, Imgproc.THRESH_BINARY);

        // 腐蚀
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
        Imgproc.erode(binaryMat, binaryMat, element);

//        // 边缘检测
//        Mat cannyMat = new Mat();
//        Imgproc.Canny(binaryMat, cannyMat, 50, 200);
//
//        // 霍夫变换检测直线
//        Mat lineMat = new Mat();
//        Imgproc.HoughLinesP(cannyMat, lineMat, 1, Math.PI/180, 50, 10, 10);
//        // 直线端点坐标
//        float sum = 0, index = 0;
//        for (int i = 0; i < lineMat.rows(); i++) {
//            double[] vec = lineMat.get(i, 0);
//            double x1 = vec[0], y1 = vec[1], x2 = vec[2], y2 = vec[3];
//            Point start = new Point(x1, y1);
//            Point end = new Point(x2, y2);
//            Imgproc.line(cannyMat, start, end, new Scalar(255, 0, 255), 1, Imgproc.LINE_AA, 0);
//            // 计算角度
//            double x3 = x2 - x1;
//            double y3 = y2 - y1;
//            double xrad = Math.atan2(y3, x3);
//            double angle = xrad / Math.PI * 180;
//
//            // 除去倾斜度为0的
//            if (angle == 0) {
//                index++;
//            }
//            sum += angle;
//            System.out.println(angle);
//        }
//
//        float average = sum / (lineMat.rows() - index); // 对所有角度求平均
//
//        // 逆时针旋转
//        Point center = new Point(mat.width() / 2, mat.height() / 2);
//        Mat affineTrans = Imgproc.getRotationMatrix2D(center, average, 1);
//        // 仿射变换，背景色填充为白色
////        Imgproc.warpAffine(mat, mat, affineTrans, mat.size(), 1, 0, new Scalar(255, 255, 255));
//        Imgproc.warpAffine(binaryMat, binaryMat, affineTrans, binaryMat.size(), Imgproc.INTER_CUBIC | Imgproc.WARP_FILL_OUTLIERS, 0, new Scalar(255, 255, 255));
//
        Imgcodecs.imwrite(IMGPATH + "new.png", binaryMat);

        imgCut(binaryMat);

        System.out.println("success");
    }

    /**
     * 图片裁剪
     * @param srcImg
     */
    private static void imgCut(Mat srcImg) {
        // 设置需要裁剪的区域
        Rect rect = new Rect(20, 20, srcImg.width() / 2, srcImg.height() / 2);
        // 感兴趣图形
        Mat roi_img = new Mat(srcImg, rect);
        // 临时图形
        Mat tmpMat = new Mat();
        // 拷贝
        roi_img.copyTo(tmpMat);
        // 输出
        Imgcodecs.imwrite(IMGPATH + "new-cut.png", tmpMat);
    }
}
