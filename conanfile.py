from conans import ConanFile, tools, CMake

class EvolveNNConan(ConanFile):
    name = "EvolveNN"
    version = "0.1"
    requires = "tbb/2020.3", "boost/1.76.0", "catch2/2.13.7"
    default_options = {
        "boost:numa": False,
        "boost:zlib": False,
        "boost:bzip2": False,
        "boost:lzma": False,
        "boost:zstd": False,
        }
    settings = "os", "compiler", "arch", "build_type"
    generators = "cmake_find_package"

    def imports(self):
        self.copy("*.dll", keep_path=False)

    def build(self):
        cmake = CMake(self, generator="Ninja")
        cmake.configure()
        cmake.build()
