-- Removes attributes from section titles that would cause Pandoc to
-- emit \hypertarget commends because \hypertarget doesn't work with
-- Alon.cls, and instead emit a raw LaTeX \label command after the
-- header.
if FORMAT:match 'latex' then
   function Header(header)
      if header.identifier then
         local identifier = header.identifier
         header.identifier = ""
         return {
            header,
            pandoc.RawBlock('latex', '\\label{' .. identifier .. "}")
         }
      end
   end
end
