mod knn;

use knn::prelude::*;

fn main() {
    let mut knn = Knn::new();
    knn.read_file("./data/iris.data");
    knn.load_data();
    knn.verify_train_accuracy(3, 10, 1000);
    knn.verify_train_accuracy(5, 10, 1000);
    knn.verify_train_accuracy(7, 10, 1000);
    knn.verify_train_accuracy(9, 10, 1000);
    knn.verify_train_accuracy(3, 20, 1000);
    knn.verify_train_accuracy(5, 20, 1000);
    knn.verify_train_accuracy(7, 20, 1000);
    knn.verify_train_accuracy(9, 20, 1000);
    knn.verify_train_accuracy(3, 30, 1000);
    knn.verify_train_accuracy(5, 30, 1000);
    knn.verify_train_accuracy(7, 30, 1000);
    knn.verify_train_accuracy(9, 30, 1000);
}
