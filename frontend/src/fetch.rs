use std::rc::Rc;

#[derive(serde::Deserialize, Clone, Debug, PartialEq)]
pub struct Classification {
    pub label: String,
    pub path: String,
    pub score: Option<f64>,
}

pub async fn generate(text: Rc<Option<String>>) -> Option<Result<Vec<Classification>, String>> {
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

pub async fn templates() -> Result<Vec<Classification>, String> {
    let response = reqwest::get("http://localhost:5000/templates")
        .await
        .unwrap();
    let templates = response.json().await.unwrap();

    Ok(templates)
}
