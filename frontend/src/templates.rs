use cobul::*;
use yew::suspense::use_future_with_deps;
use yew::*;

use crate::fetch;
use crate::fetch::Score;
use crate::image::{DynImage, Height};

#[derive(Properties, Clone, PartialEq)]
pub struct TemplateProps {
    pub format: String,
}

#[function_component(Template)]
pub fn template(props: &TemplateProps) -> Html {
    let src = format!("templates/{}", &props.format);
    let class = classes!("column", "has-text-centered", "is-one-fifth");

    let text = props.format.split('.').next().unwrap();
    let text = text.replace('-', " ");

    html! {
        <div {class} >
            <DynImage {src} height={Height::Px(300)} border=false />
            <p> {text} </p>
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
        { for scores.iter().map(|Score{label, score}| html! {<p> {label} {":"} {score} </p>}) }
    };

    Ok(inner)
}
