% Script to list files in a directory
directory_path = 'F:/RICA/metaheuristic-manifold-optimization-main (1)/metaheuristic-manifold-optimization-main/data'; % Change this to your desired path

% Check if directory exists
if ~isfolder(directory_path)
    fprintf('Directory "%s" not found\n', directory_path);
    return;
end

% Get directory contents
files = dir(directory_path);

% Display file names (excluding . and ..)
fprintf('Files in %s:\n', directory_path);
for i = 1:length(files)
    if ~strcmp(files(i).name, '.') && ~strcmp(files(i).name, '..')
        fprintf('%s\n', files(i).name);
    end
end