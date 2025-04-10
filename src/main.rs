mod knn;

use knn::prelude::*;

fn main() {
    let mut knn = Knn::new();
    knn.read_file("/home/jhin/Downloads/iris/iris.data");
    knn.load_data(10);
    let testar = Line {
        sepal_length: 4.6,
        sepal_width: 1.8,
        petal_length: 4.4,
        petal_width: 1.6,
        class: Classes::Virginica, // Just to test
    };
    let guess = knn.guess_class(&testar, 7);
    println!("Valor Real: {:?}, Valor Gerado: {:?}", testar.class, guess);
    knn.verify_train_accuracy(3);
}
