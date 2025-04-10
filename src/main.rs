mod knn;

use knn::prelude::*;

fn main() {
    let mut knn = Knn::new();
    knn.read_file("/home/jhin/Downloads/iris/iris.data");
    knn.load_data(30);
    knn.verify_train_accuracy(5);
}
