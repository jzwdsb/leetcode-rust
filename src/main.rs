pub mod basic_types;
pub mod stack;
pub mod strings;
pub mod list;
pub mod array;

pub mod dynamic_programming;
pub mod search;

// pub mod http;
// pub mod postgres;

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
