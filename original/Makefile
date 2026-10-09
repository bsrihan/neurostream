.PHONY: format export_env

format:
	# remove unused imports
	autoflake --in-place --remove-all-unused-imports \
		--ignore-init-module-imports --ignore-pass-after-docstring -r *
	# sort imports
	isort . --skip lib
	# format with YAPF
	yapf -irp . --exclude lib

export_env:
	echo "Exporting Python environment"
	conda env export --no-build | grep -v '^prefix:' > environment.yml
	pip list --exclude-editable --format=freeze > requirements.txt