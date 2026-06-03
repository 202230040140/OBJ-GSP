#pragma once

#include <string>

struct RuntimeConfig {
	std::string data_root = "./input-data";
	std::string graph_root = "";
	std::string sam_root = "";
	std::string output_root = "./input-data";
};

extern RuntimeConfig g_runtime_config;

std::string normalizePath(std::string path);
std::string joinPath(const std::string& left, const std::string& right);
void ensureDirectory(const std::string& path);
