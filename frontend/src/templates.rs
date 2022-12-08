use cobul::*;
use yew::suspense::use_future_with_deps;
use yew::*;

use crate::fetch;
use crate::fetch::Classification;
use crate::image::{DynImage, Height};

#[derive(Properties, Clone, PartialEq)]
pub struct TemplateProps {
    pub path: String,
    pub label: String,

    #[prop_or_default]
    pub score: Option<f64>,
}

#[function_component(Template)]
pub fn template(props: &TemplateProps) -> Html {
    let class = classes!("column", "has-text-centered", "is-one-fifth");

    let score = match props.score {
        Some(score) if score < 0.001 => "<0.1%".to_owned(),
        Some(score) => format!("{:.1}%", score * 100.0),
        None => String::new(),
    };

    html! {
        <div {class}>
            <DynImage src={props.path.clone()} height={Height::Px(300)} border=false />
            <b> {&props.label} </b>
            <p> {score} </p>
        </div>
    }
}

#[derive(Properties, Clone, PartialEq)]
pub struct TemplatesProps {
    pub caption: Option<String>,
}

#[function_component(Templates)]
pub fn templates(props: &TemplatesProps) -> HtmlResult {
    let caption = props.caption.clone();
    let handle = use_future_with_deps(|rc| fetch::generate(rc), caption)?;

    let scores = match (*handle).clone() {
        Some(scores) => scores.unwrap(),
        None => return Ok(html! {}),
    };

    // let ids = (*handle).clone().unwrap();

    // Ok(html! { ids[0].clone() })

    // let inner = html! {
    //     <Columns>
    //     <Template format="10-Guy.jpg" />
    //     </Columns>
    // };

    let inner = html! {
        <Columns>
        { for scores.into_iter().map(|x| html! {<Template path={x.path} label={x.label} score={x.score} />}) }
        </Columns>
    };

    Ok(inner)
}
