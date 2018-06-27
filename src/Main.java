import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {

    private static final String IMG = System.getProperty("user.dir") + "\\Img\\test.png";
    private static final String IMGPATH = System.getProperty("user.dir") + "\\Img\\";

    public static void main(String[] args) {
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

        // 边缘检测
        Mat cannyMat = new Mat();
        Imgproc.Canny(binaryMat, cannyMat, 50, 200);

        // 霍夫变换检测直线
        Mat lineMat = new Mat();
        Imgproc.HoughLinesP(cannyMat, lineMat, 1, Math.PI/180, 50, 10, 10);
        // 直线端点坐标
        float sum = 0;
        for (int i = 0; i < lineMat.rows(); i++) {
            double[] vec = lineMat.get(i, 0);
            double x1 = vec[0], y1 = vec[1], x2 = vec[2], y2 = vec[3];
            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);
            Imgproc.line(cannyMat, start, end, new Scalar(255, 0, 255), 1, Imgproc.LINE_AA, 0);
            // 计算角度
            double x3 = x2 - x1;
            double y3 = y2 - y1;
            double xrad = Math.atan2(y3, x3);
            double angle = xrad / Math.PI * 180;

            sum += angle;
            System.out.println(angle);
        }

        float average = sum / lineMat.rows(); // 对所有角度求平均

        // 逆时针旋转
        Point center = new Point(mat.width() / 2, mat.height() / 2);
        Mat affineTrans = Imgproc.getRotationMatrix2D(center, average, 1);
        // 仿射变换，背景色填充为白色
//        Imgproc.warpAffine(mat, mat, affineTrans, mat.size(), 1, 0, new Scalar(255, 255, 255));
        Imgproc.warpAffine(binaryMat, binaryMat, affineTrans, binaryMat.size(), Imgproc.INTER_CUBIC | Imgproc.WARP_FILL_OUTLIERS, 0, new Scalar(255, 255, 255));

        Imgcodecs.imwrite(IMGPATH + "new.png", binaryMat);

        System.out.println("success");
    }
}
