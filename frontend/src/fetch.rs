use std::rc::Rc;

#[derive(serde::Deserialize, Clone, Debug, PartialEq)]
pub struct Score {
    pub label: String,
    pub score: f64,
}

pub async fn generate(text: Rc<Option<String>>) -> Option<Result<Vec<Score>, String>> {
    let text = match text.as_ref() {
        Some(text) => text,
        None => return None,
    };

    let url = format!("http://localhost:5000/generate");
    let response = reqwest::Client::new()
        .post(&url)
        .body(text.clone())
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    Some(Ok(response))
}

pub async fn templates() -> Result<Vec<String>, String> {
    let response = reqwest::get(format!("http://localhost:5000/templates"))
        .await
        .unwrap();
    let templates: Vec<String> = response.json().await.unwrap();

    Ok(templates)
}
