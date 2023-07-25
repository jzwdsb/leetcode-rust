mod math;
mod stack;
mod strings;
mod list;
mod array;
mod tree;
mod graph;

mod dynamic_programming;
mod search;
mod backtrack;
mod recursive;
mod sort;
mod sliding_window;
// mod http;
// mod postgres;

mod cache;

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
