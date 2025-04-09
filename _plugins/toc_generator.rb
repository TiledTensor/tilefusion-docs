require 'nokogiri'

module Jekyll
  class TocGenerator < Generator
    safe true
    priority :low

    def generate(site)
      site.pages.each do |page|
        # Skip non-markdown files
        next unless page.ext == '.md' || page.ext == '.markdown'

        # Skip specific files and directories
        next if skip_file?(page)

        # Skip if the page already has a TOC (contains a list at the beginning)
        content = page.content
        next if content =~ /^\s*[-*]\s+\[.*?\]\(#.*?\)/m

        # Extract headings from the content
        headings = extract_headings(content)
        next if headings.empty?

        # Generate TOC
        toc = generate_toc(headings)

        # Insert TOC after the front matter
        if content =~ /^---\n.*?\n---\n/m
          front_matter = $&
          rest_of_content = $'
          page.content = front_matter + "\n" + toc + "\n\n" + rest_of_content
        else
          page.content = toc + "\n\n" + content
        end
      end
    end

    private

    def skip_file?(page)
      # Files to skip
      skip_files = [
        'index.md',
        'README.md',
        'LICENSE.md'
      ]

      # Directories to skip
      skip_dirs = [
        '_includes',
        '_layouts',
        'assets'
      ]

      # Check if file should be skipped
      return true if skip_files.include?(File.basename(page.path))
      return true if skip_dirs.any? { |dir| page.path.start_with?(dir) }

      # Skip if front matter explicitly sets no_toc: true
      return true if page.data['no_toc'] == true

      false
    end

    def extract_headings(content)
      headings = []
      content.each_line do |line|
        if line =~ /^(#+)\s+(.+)$/
          level = $1.length
          text = $2.strip
          id = text.downcase.gsub(/[^\w\s-]/, '').gsub(/\s+/, '-')
          headings << { level: level, text: text, id: id }
        end
      end
      headings
    end

    def generate_toc(headings)
      toc = []

      # Generate TOC entries
      headings.each do |heading|
        indent = "  " * (heading[:level] - 1)
        toc << "#{indent}- [#{heading[:text]}](##{heading[:id]})"
      end

      toc.join("\n")
    end
  end
end
