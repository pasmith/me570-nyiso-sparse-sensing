-- Render placeholder chits as colored tcolorbox callouts in LaTeX/PDF output.
-- A fenced div  ::: {.figbox}  or  ::: {.tablebox}  is wrapped in the matching
-- environment defined in pandoc-header.tex; its content stays Markdown.
function Div(el)
  local env
  if el.classes:includes('figbox') then
    env = 'placeholderfig'
  elseif el.classes:includes('tablebox') then
    env = 'placeholdertable'
  else
    return nil
  end
  local out = { pandoc.RawBlock('latex', '\\begin{' .. env .. '}') }
  for _, b in ipairs(el.content) do out[#out + 1] = b end
  out[#out + 1] = pandoc.RawBlock('latex', '\\end{' .. env .. '}')
  return out
end
