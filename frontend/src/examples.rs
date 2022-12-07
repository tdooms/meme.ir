use cobul::*;
use yew::*;

#[derive(Clone, PartialEq, Properties)]
struct ExampleProps {
    pub lines: Vec<&'static str>,
    pub click: Callback<()>,
}

#[derive(Clone, PartialEq, Properties)]
pub struct ExamplesProps {
    pub input: Callback<String>,
}

#[function_component(Example)]
fn example(props: &ExampleProps) -> Html {
    let hovered = use_state(|| false);
    let color = hovered.then(|| "has-background-white-ter");
    let class = classes!("box", color);

    let onmouseenter = {
        let hovered = hovered.clone();
        Callback::from(move |_| hovered.set(true))
    };
    let onmouseleave = {
        let hovered = hovered.clone();
        Callback::from(move |_| hovered.set(false))
    };

    html! {
        <Column>
            <div {class} onclick={props.click.reform(|_| ())} style="height:100%;cursor:pointer" {onmouseleave} {onmouseenter}>
            {for props.lines.iter().map(|line| html! {<p> {line} </p>})}
            </div>
        </Column>
    }
}

#[function_component(Examples)]
pub fn examples(props: &ExamplesProps) -> Html {
    let examples = [
        vec!["One does not simply generate memes with AI"],
        vec![
            "Programming in Java",
            "Programming in C",
            "Programming in Assembly",
            "Programming in Rust",
        ],
        vec!["Putin", "Putin", "Russian voters"],
    ];

    let input = props.input.clone();
    let input = |lines: &Vec<&'static str>| {
        let str = lines.join(". ");
        input.reform(move |_| str.clone())
    };

    html! {
        <Columns>
        {for examples.iter().map(|lines| html! {<Example lines={lines.clone()} click={input(lines)}/>})}
        </Columns>
    }
}
