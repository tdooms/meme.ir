use cobul::*;
use yew::suspense::{use_future, use_future_with_deps};
use yew::*;

mod examples;
mod fetch;
mod image;
mod templates;

use examples::*;
use templates::*;

pub enum Page {
    Home,
    Labeling,
}

#[function_component(Home)]
fn home() -> Html {
    let model = use_model(|| String::new());
    let state = use_state(|| None);

    let input = {
        let model = model.clone();
        Callback::from(move |value: String| model.input.emit(value))
    };

    let click = {
        let value = Some(model.value.clone());
        let state = state.clone();
        Callback::from(move |_| state.set(value.clone()))
    };

    let fallback = html! { <Progress color={Color::Success}/> };

    html! {
        <>
        <Hero color={Color::Success} size={HeroSize::Small}>
            <Title> {"meme.ir"} </Title>
            <Subtitle> {"Generate memes from only text"} </Subtitle>
        </Hero>

        <Block/>

        <Field addons=true>
            <Control expanded=true> <Input {model} size={Size::Large} /> </Control>
            <Control> <simple::Button text="generate" icon={fa::Solid::Gears} color={Color::Success} size={Size::Large} {click} /> </Control>
        </Field>

        <Examples {input} />

        <Suspense {fallback}>
        <Templates caption={(*state).clone()} />
        </Suspense>

        <Box>
            <Block> <Icon icon={fa::Brands::Github}/> <a href={"https://github.com/tdooms/meme.ir"}> {"repository"} </a> </Block>
            <Block> <Icon icon={fa::Solid::Database}/> <a href={"https://github.com/schesa/ImgFlip575K_Dataset"}> {"dataset"} </a> </Block>
        </Box>
        </>
    }
}

#[function_component(Labeling)]
fn labeling() -> HtmlResult {
    let handle = use_future(fetch::templates)?;
    let templates = (*handle).clone().unwrap();

    Ok(html! {
        <Columns multiline=true>
        { for templates.into_iter().map(|x| html! { <Template path={x.path} label={x.label} /> }) }
        </Columns>
    })
}

#[function_component(App)]
pub fn app() -> Html {
    let page = use_state(|| Page::Home);

    let inner = match *page {
        Page::Home => html! { <Home /> },
        Page::Labeling => html! { <Labeling /> },
    };

    html! { <main> <Section> {inner} </Section> </main> }
}

fn main() {
    wasm_logger::init(wasm_logger::Config::new(log::Level::Trace));
    Renderer::<App>::new().render();
}
