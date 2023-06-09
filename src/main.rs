pub mod math;
pub mod stack;
pub mod strings;
pub mod list;
pub mod array;
pub mod tree;

pub mod dynamic_programming;
pub mod search;
pub mod backtrack;
pub mod recursive;
pub mod sort;
pub mod sliding_window;
// pub mod http;
// pub mod postgres;

use clap::Parser;


#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    function_name: Option<String>
}

fn main() {
    let args = Args::parse();
    match args.function_name {
        Some(function_name) => {
            match function_name.as_str() {
                "basic_types" => math::main(),
                "stack" => stack::main(),
                "strings" => strings::main(),
                "list" => list::main(),
                "array" => array::main(),
                "dynamic_programming" => dynamic_programming::main(),
                "search" => search::main(),
                "backtrack" => backtrack::main(),
                "recursive" => recursive::main(),
                "sort" => sort::main(),
                _ => println!("No function found with name {}", function_name)
            }
        },
        None => println!("No function name provided")
    }
}
