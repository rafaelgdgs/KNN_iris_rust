pub mod prelude {
    pub use super::Knn;
}

pub struct Knn {
    file_path: String,
    file_content: Vec<String>,
    dataset: Vec<Line>,
    test_dataset: Vec<Line>,
}

impl Knn {
    pub fn new() -> Self {
        Self {
            file_path: "".to_string(),
            file_content: vec![],
            dataset: vec![],
            test_dataset: vec![],
        }
    }

    pub fn read_file(&mut self, path: &str) {
        self.file_path = path.to_string();
        self.file_content = Knn::prepare_data(
            std::fs::read_to_string(path).expect("Couldn't read the file."),
            '\n',
        );
    }

    fn prepare_data(data: String, delim: char) -> Vec<String> {
        data.split(delim).map(|x| x.to_string()).collect()
    }

    pub fn load_data(&mut self, k: usize) {
        assert!(
            (0..=100).contains(&k),
            "\"k\" not in the corrent range: 0 to 100."
        );
        let mut data: Vec<Line> = vec![];
        for line in &self.file_content {
            if line.is_empty() {
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
            let class = Knn::classify_class(parts.next().expect("Failed to get Petal Width"))
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

        use rand::seq::SliceRandom;

        data.shuffle(&mut rand::rng());
        let (test, train) = data.split_at(k);
        self.test_dataset = test.to_vec();
        self.dataset = train.to_vec();
    }

    pub fn guess_class(&self, detect: &Line, k: usize) -> Classes {
        use std::collections::HashMap;

        let mut distances: Vec<(f32, usize)> = vec![];
        for (index, item) in self.dataset.iter().enumerate() {
            let dist1 = (item.sepal_length - detect.sepal_length).abs();
            let dist2 = (item.sepal_width - detect.sepal_width).abs();
            let dist3 = (item.petal_length - detect.petal_length).abs();
            let dist4 = (item.petal_width - detect.petal_width).abs();
            distances.push((dist1 + dist2 + dist3 + dist4, index));
        }

        distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect("Failed to order the distances vector.")
        });

        let k_elements: Vec<(f32, usize)> = distances.into_iter().take(k).collect();
        let mut hash_counter: HashMap<Classes, i32> = HashMap::new();
        for &valor in &k_elements {
            *hash_counter
                .entry(self.dataset[valor.1].class.clone())
                .or_insert(0) += 1;
        }

        hash_counter
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(valor, _)| valor)
            .expect("Failed to get most frequent one")
    }

    pub fn verify_train_accuracy(&self, k: usize) {
        let total = self.test_dataset.len();
        let mut correct: f32 = 0.;

        for item in &self.test_dataset {
            let guessed = self.guess_class(item, k);
            if guessed == item.class {
                correct += 1.;
            }
        }
        println!(
            "Test accuracy: {}/{} -> {}%",
            correct,
            total,
            (correct / total as f32) * 100.
        )
    }

    fn classify_class(str: &str) -> Option<Classes> {
        match str {
            "Iris-virginica" => Some(Classes::Virginica),
            "Iris-versicolor" => Some(Classes::Versicolour),
            "Iris-setosa" => Some(Classes::Setosa),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Classes {
    Setosa,
    Versicolour,
    Virginica,
}

#[derive(Debug, Clone)]
pub struct Line {
    pub sepal_length: f32,
    pub sepal_width: f32,
    pub petal_length: f32,
    pub petal_width: f32,
    pub class: Classes,
}
