use bindgen;
use cmake;
use std::env;

//FIXME: complaining about missing struct tags in bindgen (even though they are typedef'd) (had to be manually changed)
//TODO: test above in a normal c project, could be a V-EZ issue

//TODO: fork V-EZ so we can fix cmake output dir to be in our build location?

fn main() {
    let out_dir = env::var("OUT_DIR").expect("failed to get OUT_DIR");

    let vk_dir = env::var("VULKAN_SDK").expect("failed to find Vulkan SDK");

    let vez_bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}\\Include\\", vk_dir))
        .header("V-EZ/Source/VEZ.h")
        .rustified_enum("*")
        .rustfmt_bindings(true)
        .layout_tests(false)
        .generate()
        .expect("Unable to generate VEZ.h bindings");

    vez_bindings
        .write_to_file("src/vez.rs")
        .expect("Couldn't write bindings!");

    println!("OUT DIR: {}", out_dir);

    let _dst = cmake::Config::new("V-EZ")
        .define("VEZ_OUTPUT_DIRECTORY", &out_dir)
        .build();

    println!("cargo:rustc-link-search={}\\Lib", vk_dir);
    println!("cargo:rustc-link-lib=vulkan-1");

    println!("cargo:rustc-link-search={}", &out_dir);
    println!("cargo:rustc-link-lib=VEZd");

    //FIXME: STATUS_DLL_NOT_FOUND in triangle.exe
    //FIX: do we need to copy the VEZd DLL into the executable directory? or add to search path
}
