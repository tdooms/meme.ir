use yew::*;
use yew_router::*;
use cobul::*;

#[fucntinon_component(Templates)]
fn templates() -> HtmlResult {
    html! {
    }
}

#[function_component(App)]
pub fn app() -> Html {
    let templates = use_state(|| None);
    let model = use_model(|| String::new());

    let cloned = templates.clone();
    let click = Callback::from(move || {

    });

    html! {
        <main>
            <BrowserRouter>
                <Section>
                    <simple::Field label={"Meme text"}>
                    <Input {model} />
                    </simple::Field>

                    <simple::Button text="generate" icon={fa::Solid::Gears} {click} />
                </Section>
            </BrowserRouter>
        </main>
    }
}

fn main() {
    wasm_logger::init(wasm_logger::Config::new(log::Level::Trace));
    Renderer::<App>::new().render();
}
