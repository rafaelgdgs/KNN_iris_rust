fn read_file(path: &str) -> String {
    std::fs::read_to_string(path).expect("Couldn't read the file.")
}

fn load_data(str: String) -> Vec<Line> {
    let mut data: Vec<Line> = vec![];
    for line in str.split('\n') {
        #[cfg(debug_assertions)]
        println!("Reading line: {}", line);
        if line == "" {
            continue;
        }
        let mut parts = line.split(',');
        let sepal_length = parts
            .next()
            .expect("Failed to get Sepal Length")
            .parse::<f32>()
            .expect("Failed to define sepal_length");
        let sepal_width = parts
            .next()
            .expect("Failed to get Sepal Width")
            .parse::<f32>()
            .expect("Failed to define sepal_width");
        let petal_length = parts
            .next()
            .expect("Failed to get Petal Length")
            .parse::<f32>()
            .expect("Failed to define petal_length");
        let petal_width = parts
            .next()
            .expect("Failed to get Petal Width")
            .parse::<f32>()
            .expect("Failed to define petal_width");
        let class = classify_class(parts.next().expect("Failed to get Petal Width"))
            .expect("Couldn't set class");
        let current: Line = Line {
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            class,
        };
        data.push(current);
    }
    data
}

fn classify_class(str: &str) -> Option<Classes> {
    match str {
        "Iris-virginica" => Some(Classes::Virginica),
        "Iris-versicolor" => Some(Classes::Versicolour),
        "Iris-setosa" => Some(Classes::Setosa),
        _ => None,
    }
}

#[derive(Debug)]
enum Classes {
    Setosa,
    Versicolour,
    Virginica,
}

#[derive(Debug)]
struct Line {
    sepal_length: f32,
    sepal_width: f32,
    petal_length: f32,
    petal_width: f32,
    class: Classes,
}

fn main() {
    let model_file_data: String = read_file("/home/jhin/Downloads/iris/iris.data");
    let model_data: Vec<Line> = load_data(model_file_data);
    println!("{:?}", model_data[0]);
}
