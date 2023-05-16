pub mod basic_types;
pub mod stack;
pub mod strings;
pub mod list;


use clap::Parser;


#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    function_name: Option<String>
}

fn main() {
    // TODO: should be able to pass in a function name to run
    //       if function name is none, run all functions
}
