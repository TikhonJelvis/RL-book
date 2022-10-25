-- Replace em-dashes with \emdash so that words before/after em-dashes
-- can be hyphenated correctly
if FORMAT:match 'latex' then
   function Str(s)
      local t = s.text
      if string.find(t, '%-%-%-') or string.find(t, '—') then
         t = string.gsub(t, '%-%-%-', '\\emdash ')
         t = string.gsub(t, '—', '\\emdash ')
         return pandoc.RawInline('tex', t)
      end
   end
end
