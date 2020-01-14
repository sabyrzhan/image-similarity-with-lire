package de.mayflower.samplecode.SimilaritySearchWithLIRE;

import net.semanticmetadata.lire.builders.GlobalDocumentBuilder;
import net.semanticmetadata.lire.imageanalysis.features.global.FCTH;
import net.semanticmetadata.lire.indexers.hashing.LocalitySensitiveHashing;
import net.semanticmetadata.lire.utils.SerializationUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.util.MathArrays;
import org.apache.lucene.document.Document;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SimilaritySearchWithLIRE {

    static class ImageInImageDatabase {

        public String fileName;
        public double[] fcthFeatureVector;
        public double distanceToSearchImage;

        @Override
        public String toString() {
            return "[FileName: " + fileName + ", dist: " + distanceToSearchImage + "]";
        }
    }

    static class ImageComparator implements Comparator<ImageInImageDatabase> {

        @Override
        public int compare(ImageInImageDatabase object1, ImageInImageDatabase object2) {
            if (object1.distanceToSearchImage < object2.distanceToSearchImage) {
                return -1;
            } else if (object1.distanceToSearchImage > object2.distanceToSearchImage) {
                return 1;
            } else {
                return 0;
            }
        }
    }

    public static double[] getFCTHFeatureVector(String fullFilePath, String id) throws FileNotFoundException, IOException {

        GlobalDocumentBuilder builder = new GlobalDocumentBuilder(true, GlobalDocumentBuilder.HashingMode.LSH);
        LocalitySensitiveHashing.readHashFunctions();
        builder.addExtractor(FCTH.class);
        BufferedImage bufferedImage = ImageIO.read(new FileInputStream(fullFilePath));
        Document doc = builder.createDocument(bufferedImage, id);

        FCTH fcthDescriptor = new FCTH();
        fcthDescriptor.setByteArrayRepresentation(doc.getFields().get(1).binaryValue().bytes);

        return fcthDescriptor.getFeatureVector();

    }

    public static double calculateEuclideanDistance(double[] vector1, double[] vector2) {

        double innerSum = 0.0;
        for (int i = 0; i < vector1.length; i++) {
            innerSum += Math.pow(vector1[i] - vector2[i], 2.0);
        }

        return Math.sqrt(innerSum);

    }

    private static double[] generate(int featureSize) {
        NormalDistribution normalDistribution = new NormalDistribution();
        double[] result = new double[featureSize];
        for(int i = 0; i < featureSize; i++) {
            result[i] = normalDistribution.sample();
        }

        return result;
    }

    private static double[][] generate2D(int hashSize, int dim) {
        NormalDistribution normalDistribution = new NormalDistribution();
        double[][] result = new double[hashSize][dim];
        for(int i = 0; i < hashSize; i++) {
            for(int j = 0; j < dim; j++) {
                result[i][j] = normalDistribution.sample();
            }
        }

        return result;
    }

    private static double[] dot(double[] featureMatrix, double[][] random) {
        double[] result = new double[random.length];
        for(int i = 0; i < random.length; i++) {
            result[i] = MathArrays.linearCombination(featureMatrix, random[i]);
        }

        return result;
    }

    private static String generateBucket(double[] result) {
        String ret = "";
        for(int i = 0; i < result.length; i++) {
            ret += result[i] > 0 ? "1" : "0";
        }

        return ret;
    }

    private static void writeRandom(double[][] values) throws Exception {
        FileOutputStream fos = new FileOutputStream("/Users/sabyrzhan/projects/image-similarity-with-lire/random.txt");
        PrintWriter printWriter = new PrintWriter(fos);
        for(int i = 0; i < values.length; i++) {
            double[] rowValues = values[i];
            String line = StringUtils.join(rowValues, ',');
            printWriter.println(line);
        }
        printWriter.close();
        fos.close();
    }

    private static double[][] readRandom() throws Exception {
        List<double[]> result = new ArrayList<>();
        FileInputStream fis = new FileInputStream("/Users/sabyrzhan/projects/image-similarity-with-lire/random.txt");
        Scanner scanner = new Scanner(fis);
        while(scanner.hasNextLine()) {
            String line = scanner.nextLine();
            if(StringUtils.isBlank(line)) {
                break;
            }

            String[] tokens = StringUtils.split(line, ',');
            double[] values = new double[tokens.length];

            for(int i = 0; i < values.length; i++) {
                values[i] = Double.valueOf(tokens[i]);
            }
            result.add(values);
        }

        return result.toArray(new double[0][0]);
    }

    public static void main(String[] args) throws Exception {
        Map<String, List<Item>> buckets = new LinkedHashMap<>();
        double[][] initialProjection = generate2D(6, 192);
        String dirName = "/Users/sabyrzhan/projects/image-similarity-with-lire/sample_images_2";
        File directory = new File(dirName);

        FilenameFilter filter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") || name.endsWith(".jpeg");
            }
        };

        String[] fileNames = directory.list(filter);

        for (String fileName : fileNames) {
            double[] fcthFeatureVector = getFCTHFeatureVector(dirName + "/" + fileName, fileName);
            double[] tmpResult = dot(fcthFeatureVector, initialProjection);
            //double distanceToSearchImage = calculateEuclideanDistance(fcthFeatureVector, searchImageFeatureVector);
            String bucketTmp = generateBucket(tmpResult);
            Item item = new Item();
            item.fileName = fileName;
            item.feature = fcthFeatureVector;

            ImageInImageDatabase imageInImageDatabase = new ImageInImageDatabase();

            imageInImageDatabase.fileName = fileName;
            imageInImageDatabase.fcthFeatureVector = fcthFeatureVector;

            item.imageInImageDatabase = imageInImageDatabase;

            if(buckets.get(bucketTmp) != null) {
                buckets.get(bucketTmp).add(item);
            } else {
                buckets.put(bucketTmp, new ArrayList<>());
                buckets.get(bucketTmp).add(item);
            }
        }

        ImageInImageDatabase mostSimilar = null;

        for(int i = 0; i < 250; i++) {
            double[] searchImageFeatureVector = getFCTHFeatureVector("/Users/sabyrzhan/projects/image-similarity-with-lire/search_linux.jpg", "11");
            double[][] projection = generate2D(6, searchImageFeatureVector.length);

            double[] dotProduct = dot(searchImageFeatureVector, projection);
            String bucket = generateBucket(dotProduct);

            if(buckets.get(bucket) != null) {
                ImageInImageDatabase tmpImage = sortAndPrintItems(buckets.get(bucket), searchImageFeatureVector);
                System.out.println(buckets.get(bucket));
                if(mostSimilar == null) {
                    mostSimilar = tmpImage;
                } else {
                    if(mostSimilar.distanceToSearchImage > tmpImage.distanceToSearchImage) {
                        mostSimilar = tmpImage;
                    }
                }
                System.out.println();
            }
        }

        System.out.println(mostSimilar);
    }

    private static ImageInImageDatabase sortAndPrintItems(List<Item> items, double[] searchFeatures) {
        List<ImageInImageDatabase> databases = new ArrayList<>();
        for(Item item: items) {
            ImageInImageDatabase imageInImageDatabase = item.imageInImageDatabase;
            imageInImageDatabase.distanceToSearchImage = calculateEuclideanDistance(searchFeatures, item.feature);
            databases.add(item.imageInImageDatabase);
        }

        Collections.sort(databases, new ImageComparator());

        System.out.println(databases);

        return databases.get(0);
    }


    private static class Item {
        private String fileName;
        private double[] feature;
        private ImageInImageDatabase imageInImageDatabase;

        @Override
        public String toString() {
            return "[" + imageInImageDatabase.fileName + ", dist: " + imageInImageDatabase.distanceToSearchImage + "]";
        }
    }
}
