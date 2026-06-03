#include "RuntimeConfig.h"

#include <filesystem>

RuntimeConfig g_runtime_config;

std::string normalizePath(std::string path) {
	for (char& ch : path) {
		if (ch == '\\') {
			ch = '/';
		}
	}
	while (path.size() > 1 && path.back() == '/') {
		path.pop_back();
	}
	return path;
}

std::string joinPath(const std::string& left, const std::string& right) {
	if (left.empty()) {
		return normalizePath(right);
	}
	if (right.empty()) {
		return normalizePath(left);
	}

	std::string lhs = normalizePath(left);
	std::string rhs = normalizePath(right);
	if (!rhs.empty() && (rhs[0] == '/' || (rhs.size() > 1 && rhs[1] == ':'))) {
		return rhs;
	}
	return lhs + "/" + rhs;
}

void ensureDirectory(const std::string& path) {
	if (!path.empty()) {
		std::filesystem::create_directories(std::filesystem::path(path));
	}
}
